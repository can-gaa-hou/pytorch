# Owner(s): ["module: PrivateUse1"]

import ctypes
import sys
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.cpp_extension import load_inline


def load_libtorch_cpu():
    if sys.platform == "win32":
        libname = "torch_cpu.dll"
    elif sys.platform == "darwin":
        libname = "libtorch_cpu.dylib"
    else:
        libname = "libtorch_cpu.so"
    return ctypes.CDLL(str(Path(torch.__file__).parent / "lib" / libname))


# Drives the AOTI runtime header paths a PrivateUse1 model.so would take:
# parse_device_str, RAII_privateuse1Malloc (constant blob allocation) and
# device_blob_memcpy (constant H2D/D2D/D2H fill). OpenReg device memory is
# mprotect(PROT_NONE)-protected, so any stray host memcpy on these paths
# crashes instead of silently writing through the wrong pointer.
CPP_SOURCE = """
#include <torch/csrc/inductor/aoti_runtime/model_base.h>

std::vector<int64_t> parse_device(std::string device_str) {
  int32_t device_type = -1;
  int32_t device_idx = -1;
  torch::aot_inductor::parse_device_str(device_str, device_type, device_idx);
  return {device_type, device_idx};
}

int64_t privateuse1_device_type() {
  return aoti_torch_device_type_privateuse1();
}

int64_t cpu_device_type() {
  return aoti_torch_device_type_cpu();
}

int64_t cuda_device_type() {
  return aoti_torch_device_type_cuda();
}

torch::Tensor blob_roundtrip(torch::Tensor src, bool via_second_blob) {
  TORCH_CHECK(src.device().is_cpu());
  TORCH_CHECK(src.is_contiguous());
  const size_t nbytes = src.nbytes();
  const int32_t pu1 = aoti_torch_device_type_privateuse1();
  const int32_t cpu = aoti_torch_device_type_cpu();
  auto blob = RAII_privateuse1Malloc(nbytes, 0);
  device_blob_memcpy(blob.get(), pu1, 0, src.data_ptr(), cpu, 0, nbytes);
  const void* read_src = blob.get();
  RAIIDataPtr second_blob;
  if (via_second_blob) {
    second_blob = RAII_privateuse1Malloc(nbytes, 0);
    device_blob_memcpy(second_blob.get(), pu1, 0, blob.get(), pu1, 0, nbytes);
    read_src = second_blob.get();
  }
  torch::Tensor out = torch::empty_like(src);
  device_blob_memcpy(out.data_ptr(), cpu, 0, read_src, pu1, 0, nbytes);
  return out;
}
"""


class TestAOTIShims(TestCase):
    """New aoti_torch_* shims backed by the OpenReg-registered accelerator."""

    @classmethod
    def setUpClass(cls):
        cls.lib = load_libtorch_cpu()

    def test_privateuse1_backend_name(self):
        fn = self.lib.aoti_torch_get_privateuse1_backend_name
        fn.restype = ctypes.c_char_p
        self.assertEqual(fn(), b"openreg")

    def test_set_get_current_device_index(self):
        get_fn = self.lib.aoti_torch_get_current_device_index
        set_fn = self.lib.aoti_torch_set_current_device_index
        idx = ctypes.c_int32(-1)
        try:
            self.assertEqual(set_fn(ctypes.c_int32(1)), 0)
            self.assertEqual(get_fn(ctypes.byref(idx)), 0)
            self.assertEqual(idx.value, 1)
            self.assertEqual(torch.accelerator.current_device_index(), 1)
        finally:
            set_fn(ctypes.c_int32(0))
        self.assertEqual(torch.accelerator.current_device_index(), 0)

    def test_synchronize_device(self):
        self.assertEqual(self.lib.aoti_torch_synchronize_device(0), 0)


class TestAOTIRuntimePrivateUse1(TestCase):
    """AOTI model.so runtime header behavior for a PrivateUse1 backend."""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_inline(
            name="aoti_runtime_openreg_test",
            cpp_sources=CPP_SOURCE,
            functions=[
                "parse_device",
                "privateuse1_device_type",
                "cpu_device_type",
                "cuda_device_type",
                "blob_roundtrip",
            ],
            with_cuda=False,
            verbose=False,
        )

    def test_parse_device_str(self):
        pu1 = self.mod.privateuse1_device_type()
        cpu = self.mod.cpu_device_type()
        cuda = self.mod.cuda_device_type()
        self.assertEqual(self.mod.parse_device("openreg"), [pu1, -1])
        self.assertEqual(self.mod.parse_device("openreg:0"), [pu1, 0])
        self.assertEqual(self.mod.parse_device("openreg:1"), [pu1, 1])
        # In-tree device strings must keep taking the regex path.
        self.assertEqual(self.mod.parse_device("cpu"), [cpu, -1])
        self.assertEqual(self.mod.parse_device("cuda:3"), [cuda, 3])

    def test_parse_device_str_invalid(self):
        bad = ["openregx", "openreg:", "openreg:x", "openreg:0x", "npu:0", ""]
        for device_str in bad:
            with self.assertRaisesRegex(RuntimeError, "Invalid device"):
                self.mod.parse_device(device_str)

    def test_constant_blob_h2d_roundtrip(self):
        # load_constants path: host constants -> device blob -> read back.
        src = torch.randn(1024)
        self.assertEqual(self.mod.blob_roundtrip(src, False), src)

    def test_constant_blob_d2d_roundtrip(self):
        # update_constant_buffer path: adds a device-to-device blob hop.
        src = torch.arange(4096, dtype=torch.int32)
        self.assertEqual(self.mod.blob_roundtrip(src, True), src)


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, 4))
        self.b = torch.nn.Parameter(torch.randn(4))

    def forward(self, x):
        # x @ w exercises proxy-executor tensor args; + 1 exercises a python
        # scalar bound to a Tensor-typed parameter (add.Tensor).
        return torch.relu(x @ self.w + self.b + 1)


class TestAOTIOpenRegE2E(TestCase):
    """Full AOTI export -> compile -> load -> run cycle on the openreg device.

    OpenReg has no compute ISA and keeps device memory PROT_NONE-protected, so
    graphs are compiled with fallback_by_default: every op dispatches through
    the proxy executor to OpenReg's registered ATen kernels, while the AOTI
    C++ wrapper and runtime handle device parsing, constant loading, buffer
    allocation, and synchronization through the PrivateUse1 runtime paths.
    """

    def _compile(self, model, args):
        ep = torch.export.export(model, args)
        with inductor_config.patch(fallback_by_default=True):
            return torch._inductor.aoti_compile_and_package(ep)

    def test_aoti_compile_load_run(self):
        model = SmallModel().eval().to("openreg")
        x = torch.randn(2, 4, device="openreg")
        package_path = self._compile(model, (x,))

        loaded = torch._inductor.aoti_load_package(package_path)
        out = loaded(x)
        self.assertEqual(out.device.type, "openreg")
        self.assertEqual(out.cpu(), model(x).cpu())

    def test_aoti_run_on_second_device(self):
        model = SmallModel().eval().to("openreg")
        x = torch.randn(2, 4, device="openreg")
        package_path = self._compile(model, (x,))

        loaded = torch._inductor.aoti_load_package(package_path, device_index=1)
        x1 = x.to("openreg:1")
        out = loaded(x1)
        self.assertEqual(out.device, torch.device("openreg", 1))
        self.assertEqual(out.cpu(), model(x).cpu())

    def test_aoti_update_constant_buffer(self):
        model = SmallModel().eval().to("openreg")
        x = torch.randn(2, 4, device="openreg")
        package_path = self._compile(model, (x,))
        loaded = torch._inductor.aoti_load_package(package_path)

        new_w = torch.randn(4, 4, device="openreg")
        new_b = torch.randn(4, device="openreg")
        loaded.loader.update_constant_buffer(
            {"w": new_w, "b": new_b}, False, False
        )
        out = loaded(x)
        expected = torch.relu(x @ new_w + new_b + 1)
        self.assertEqual(out.cpu(), expected.cpu())


class TestInductorCompileOpenReg(TestCase):
    def test_torch_compile_inductor_backend(self):
        model = SmallModel().eval().to("openreg")
        x = torch.randn(2, 4, device="openreg")
        with inductor_config.patch(fallback_by_default=True):
            compiled = torch.compile(model, backend="inductor")
            out = compiled(x)
        self.assertEqual(out.device.type, "openreg")
        self.assertEqual(out.cpu(), model(x).cpu())


if __name__ == "__main__":
    run_tests()

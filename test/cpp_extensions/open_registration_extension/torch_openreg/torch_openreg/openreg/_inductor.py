"""Inductor codegen support for the openreg device.

OpenReg is a CPU-simulated accelerator with no compute ISA of its own, so it
reuses the CPU C++ codegen classes. Direct pointer access from generated
kernels would trip OpenReg's PROT_NONE device-memory protection, so graphs
should be compiled with torch._inductor.config.fallback_by_default=True: every
op then dispatches through ATen and lands on OpenReg's registered kernels,
while Inductor still drives buffer planning and (for AOTI) the C++ wrapper.

Imported lazily via torch_openreg.openreg.__getattr__: Inductor's
init_backend_registration looks up the Scheduling / *WrapperCodegen attributes
on the registered device module the first time codegen runs.
"""

from torch._inductor.codegen.common import register_device_op_overrides
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.codegen.cpu_device_op_overrides import CpuDeviceOpOverrides
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen


class OpenRegCppWrapperCodegen(CppWrapperCpu):
    def __init__(self):
        # Must be set before CppWrapperCpu.__init__, which defaults to "cpu".
        self.device = "openreg"
        super().__init__()

    # CppWrapperCpu.create is a staticmethod hardcoded to CppWrapperCpu(),
    # which would silently drop this subclass.
    @staticmethod
    def create(is_subgraph, subgraph_name, parent_wrapper, partition_signatures=None):
        return OpenRegCppWrapperCodegen()


register_device_op_overrides("openreg", CpuDeviceOpOverrides())

Scheduling = CppScheduling
CppWrapperCodegen = OpenRegCppWrapperCodegen
__all__ = [
    "CppWrapperCodegen",
    "PythonWrapperCodegen",
    "Scheduling",
    "WrapperFxCodegen",
]

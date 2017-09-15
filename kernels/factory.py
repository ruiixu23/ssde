from kernels.linear import LinearKernel
from kernels.locally_periodic import LocallyPeriodicKernel
from kernels.periodic import PeriodicKernel
from kernels.polynomial import Polynomial
from kernels.rbf import RBFKernel
from kernels.sigmoid import SigmoidKernel


class KernelFactory:
    __factories = {
        'linear': LinearKernel,
        'locally_periodic': LocallyPeriodicKernel,
        'periodic': PeriodicKernel,
        'polynomial': Polynomial,
        'rbf': RBFKernel,
        'sigmoid': SigmoidKernel
    }
    __instances = dict()

    @staticmethod
    def supported_kernels():
        """
        Get the names of the supported kernels.

        :return: (list) The names of the supported kernels as a list of strings.
        """
        return sorted(KernelFactory.__factories.keys())

    @staticmethod
    def get_kernel(name):
        """
        Get the kernel instance with the specified name.

        :param name: str
            The name of the kernel.
        :return: Kernel
            The kernel instance.
        :raise: ValueError
            If an invalid name is used.
        """
        if name not in KernelFactory.__factories:
            raise ValueError('No kernel with name {} found.'.format(name))
        elif name not in KernelFactory.__instances:
            KernelFactory.__instances[name] = KernelFactory.__factories[name]()
            return KernelFactory.__instances[name]
        else:
            return KernelFactory.__instances[name]


import pint

# A global unit registry that can be used by any of other module.
unit_registry = pint.UnitRegistry(system="mks", autoconvert_offset_to_baseunit=True)
u = unit_registry

# default formatting includes 4 significant digits.
# This can be overridden on a per-print basis with
# print('{:.3f}'.format(3 * ureg.m / 9)).
u.default_format = ".4g"

u.define("dollar = [money] = USD")


def set_sig_figs(n=4):
    """Set the default number of significant figures used to print Pint,
    Pandas and NumPy value quantities.

    Parameters
    ----------
    n : int
        number of significant figures to display. Defaults to 4.
    """
    u.default_format = "." + str(n) + "g"

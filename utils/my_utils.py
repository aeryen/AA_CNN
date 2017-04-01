import os
import types
import time
import datetime


def get_import_modules():
    for name, val in list(globals().items()):
        if isinstance(val, types.ModuleType):
            yield val.__name__


def write_hyperparams(out_dir, FLAGS):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    hyper_param_file = open(os.path.join(out_dir, 'param.txt'), 'w')
    hyper_param_file.write("Time:\n" + str(datetime.datetime.now()) + "\n\n")
    hyper_param_file.write("Modules:\n\n")
    for mod_name in get_import_modules():
        hyper_param_file.write(mod_name + "\n")
    hyper_param_file.write("Parameters:\n")
    for attr, value in sorted(FLAGS.__flags.items()):
        hyper_param_file.write("{}={}\n".format(attr.upper(), value))
    hyper_param_file.close()

import pykeops
pykeops.verbose = True
pykeops.build_type = 'Debug'
pykeops.clean_pykeops()
pykeops.test_torch_bindings()

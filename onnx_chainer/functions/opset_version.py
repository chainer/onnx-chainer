def support(opset_versions=None):
    def _wrapper(func):
        def _func_with_lower_opset_version(*args, **kwargs):
            if opset_versions is None:
                return func(*args, **kwargs)
            opset_version = args[1]
            for opver in sorted(opset_versions, reverse=True):
                if opver <= opset_version:
                    break
            opset_version = opver
            return func(args[0], opset_version, *args[2:], **kwargs)
        return _func_with_lower_opset_version
    return _wrapper

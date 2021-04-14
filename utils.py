import epynet as en

time_params = {
    '0': 'EN_DURATION',
    '1': 'EN_HYDSTEP',
    '2': 'EN_QUALSTEP',
    '3': 'EN_PATTERNSTEP',
    '4': 'EN_PATTERNSTART',
    '5': 'EN_REPORTSTEP',
    '6': 'EN_REPORTSTART',
    '7': 'EN_RULESTEP',
    '8': 'EN_STATISTIC',
    '9': 'EN_PERIODS',
    '10': 'EN_STARTTIME'
}


def get_time_param_code(param: str):
    for key, value in time_params.items():
        if value == param:
            return int(key)
    raise KeyError("The selected parameter doesn't exist")


def get_time_parameter(wds: en.Network, code: int):
    return time_params[str(code)], en.epanet2.EPANET2.ENgettimeparam(wds.ep, code)


def time_parameters_summary(wds: en.Network):
    for i in time_params.keys():
        print(get_time_parameter(wds, int(i)))


if __name__ == '__main__':
    import utils
    print(utils.get_time_param_code("EN_HYSTEP"))



import epynet as en

time_params = {
    '1': 'EN_DURATION',
    '2': 'EN_HYDSTEP',
    '3': 'EN_QUALSTEP',
    '4': 'EN_PATTERNSTEP',
    '5': 'EN_PATTERNSTART',
    '6': 'EN_REPORTSTEP',
    '7': 'EN_REPORTSTART',
    '8': 'EN_RULESTEP',
    '9': 'EN_STATISTIC',
    '10': 'EN_PERIODS'
}


def get_time_parameter(wds: en.Network, code: int):
    return time_params[str(code)], en.epanet2.EPANET2.ENgettimeparam(wds.ep, code)


def time_parameters_summary(wds: en.Network):
    for i in range(1, 11):
        print(get_time_parameter(wds, i))


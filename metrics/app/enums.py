from enum import Enum, auto


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class MobilityAnalysisRoutesTravelTypeEnum(str, Enum):
    DRIVE = "drive"
    WALK = "walk"


class CitiesEnum(str, Enum):
    SAINT_PETERSBURG = "Saint_Petersburg"
    KRASNODAR = "Krasnodar"
    SEVASTOPOL = "Sevastopol"


class MobilityAnalysisIsochronesTravelTypeEnum(str, Enum):
    def __new__(cls, value, label):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.label = label
        return obj

    PUBLIC_TRANSPORT = "public_transport", "Общественный транспорт"
    DRIVE = "drive", "Личный транспорт"
    WALK = "walk", "Пешком"


class MobilityAnalysisIsochronesTravelTypeLabelEnum(str, Enum):
    PUBLIC_TRANSPORT = "Общественный транспорт"
    DRIVE = "Личный транспорт"
    WALK = "Пешком"


class MobilityAnalysisIsochronesWeightTypeEnum(str, Enum):
    TIME = "time"
    METER = "meter"


class InstagramConcentrationSeason(str, AutoName):
    white_nights = auto()
    summer = auto()
    winter = auto()
    spring_and_autumn = "spring+autumn"


class InstagramConcentrationDayTime(str, AutoName):
    dark = auto()
    light = auto()


class ProvisionTypeEnum(str, AutoName):
    calculated = auto()
    normative = auto()


class ProvisionLoadOptionEnum(Enum):  # todo int inheritance
    ALL_SERVICES = None
    FULLY_LOADED = 0
    MARGIN_OF_CAPACITY = 1


class ProvisionOptionEnum(Enum):  # todo int inheritance
    ALL_HOUSES = None
    DONT_FULLY_PROVIDED = 0
    FULLY_PROVIDED = 1


class ProvisionGetInfoObjectTypeEnum(str, AutoName):
    house = auto()
    service = auto()

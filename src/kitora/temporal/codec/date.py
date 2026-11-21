from dataclasses import dataclass

_DATE_FORMATS: tuple[str, ...] = ("iso", "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d")


@dataclass(frozen=True)
class DateCodecSettings:
    parse_patterns: tuple[str, ...] = _DATE_FORMATS
    """解析日期字符串 `strptime` 的格式候选"""
    


class DateCodec:
    pass

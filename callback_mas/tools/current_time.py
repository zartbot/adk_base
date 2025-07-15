import datetime
async def get_current_time() -> str:
    """
    获取当前时间
    Returns:
        str: 当前的时间, 格式为：2023-05-01 12:00:00
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


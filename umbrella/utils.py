class TextColors:
    """
    定义颜色类，用于存储 ANSI 转义序列的颜色代码。
    """
    COLORS = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m",
    }

    @staticmethod
    def colorize(text, color):
        """
        将文本着色为指定颜色。
        
        :param text: 要着色的字符串
        :param color: 颜色名称（如 "red", "blue"）
        :return: 带颜色的字符串
        """
        color_code = TextColors.COLORS.get(color.lower(), TextColors.COLORS["reset"])
        return f"{color_code}{text}{TextColors.COLORS['reset']}"
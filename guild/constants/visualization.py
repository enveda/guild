"""
Visualization constants
"""

import seaborn as sns

"""
Colors
"""
GUILD_COLORS = {
    "white": "#FFFFFF",
    "black": "#000000",
    "deep-pink": "#F10A84",
    "yellow-green": "#BCD20B",
    "slate-blue": "#6942D9",
    "khakhi": "#F6E547",
    "cornflower-blue": "#5C7CFC",
    "orange-red": "#FA5F0D",
    "olive-drab": "#6A8D3E",
    "crimson": "#E93848",
    "sky-blue": "#80BDE9",
}

CONTRAST_COLORS = {
    "bright-red": "#FF4B4B",
    "royal-blue": "#4169E1",
    "lime-green": "#32CD32",
    "dark-orchid": "#9932CC",
    "dark-orange": "#FF8C00",
    "light-sea-green": "#20B2AA",
    "crimson": "#DC143C",
    "steel-blue": "#4682B4",
    "hot-pink": "#FF69B4",
    "forest-green": "#228B22",
}
CONTRAST_PALETTE = sns.color_palette(list(CONTRAST_COLORS.values()))

GUILD_PALETTE_RAINBOW = sns.color_palette(
    [v for k, v in GUILD_COLORS.items() if k not in ["white", "black"]]
)
GUILD_PALETTE_RAINBOW_PLOTLY = [
    f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in GUILD_PALETTE_RAINBOW
]

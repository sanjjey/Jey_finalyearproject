"""
AGED Framework Configuration
Aspect-Global Evaluative Dissonance in Online Product Reviews
"""

# Default aspect taxonomy with keywords and synonyms
DEFAULT_ASPECT_TAXONOMY = {
    "battery": [
        "battery", "battery life", "charging", "charger", "drain", "charge",
        "standby", "screen on time", "sot", "power", "backup"
    ],
    "camera": [
        "camera", "photo", "photos", "picture", "pictures", "lens", "sensor",
        "video", "zoom", "portrait", "night mode", "selfie", "low light", "megapixels"
    ],
    "display": [
        "display", "screen", "oled", "amoled", "refresh rate", "hz", "brightness",
        "resolution", "bezel", "colors", "hdr"
    ],
    "performance": [
        "performance", "speed", "processor", "chipset", "lag", "ram", "multitasking",
        "gaming", "smooth", "fps", "heating", "warm", "throttle"
    ],
    "build_quality": [
        "build", "design", "durability", "premium", "plastic", "metal", "glass",
        "ergonomics", "weight", "finish", "scratches", "sturdy", "feel"
    ],
    "price_value": [
        "price", "value", "cost", "expensive", "cheap", "worth", "money",
        "budget", "affordable", "overpriced", "deal"
    ],
    "software": [
        "software", "os", "ui", "updates", "bugs", "bloatware", "interface",
        "android", "ios", "glitch", "crash"
    ]
}

# Rating scale constants
MIN_RATING = 1.0
MAX_RATING = 5.0

# Temporal window options
DEFAULT_WINDOW_DAYS = 30

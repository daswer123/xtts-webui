class Language():
    def __init__(self, code, name):
        self.code = code
        self.name = name

    def __str__(self):
        return "Language(code={}, name={})".format(self.code, self.name)

LANGUAGES = [
    Language('en', 'English'),
    Language('zh', 'Chinese'),   
    Language('de', 'German'),    
    Language('es', 'Spanish'),   
    Language('ru', 'Russian'),   
    Language('ko', 'Korean'),    
    Language('fr', 'French'),    
    Language('ja', 'Japanese'),  
    Language('pt', 'Portuguese'),
    Language('tr', 'Turkish'),   
    Language('pl', 'Polish'),    
    Language('ca', 'Catalan'),   
    Language('nl', 'Dutch'),     
    Language('ar', 'Arabic'),
    Language('sv', 'Swedish'),
    Language('it', 'Italian'),
    Language('id', 'Indonesian'),
    Language('hi', 'Hindi'),
    Language('fi', 'Finnish'),
    Language('vi', 'Vietnamese'),
    Language('he', 'Hebrew'),
    Language('uk', 'Ukrainian'),
    Language('el', 'Greek'),
    Language('ms', 'Malay'),
    Language('cs', 'Czech'),
    Language('ro', 'Romanian'),
    Language('da', 'Danish'),
    Language('hu', 'Hungarian'),
    Language('ta', 'Tamil'),
    Language('no', 'Norwegian'),
    Language('th', 'Thai'),
    Language('ur', 'Urdu'),
    Language('hr', 'Croatian'),
    Language('bg', 'Bulgarian'),
    Language('lt', 'Lithuanian'),
    Language('la', 'Latin'),
    Language('mi', 'Maori'),
    Language('ml', 'Malayalam'),
    Language('cy', 'Welsh'),
    Language('sk', 'Slovak'),
    Language('te', 'Telugu'),
    Language('fa', 'Persian'),
    Language('lv', 'Latvian'),
    Language('bn', 'Bengali'),
    Language('sr', 'Serbian'),
    Language('az', 'Azerbaijani'),
    Language('sl', 'Slovenian'),
    Language('kn', 'Kannada'),
    Language('et', 'Estonian'),
    Language('mk', 'Macedonian'),
    Language('br', 'Breton'),
    Language('eu', 'Basque'),
    Language('is', 'Icelandic'),
    Language('hy', 'Armenian'),
    Language('ne', 'Nepali'),
    Language('mn', 'Mongolian'),
    Language('bs', 'Bosnian'),
    Language('kk', 'Kazakh'),
    Language('sq', 'Albanian'),
    Language('sw', 'Swahili'),
    Language('gl', 'Galician'),
    Language('mr', 'Marathi'),
    Language('pa', 'Punjabi'),
    Language('si', 'Sinhala'),
    Language('km', 'Khmer'),
    Language('sn', 'Shona'),
    Language('yo', 'Yoruba'),
    Language('so', 'Somali'),
    Language('af', 'Afrikaans'),
    Language('oc', 'Occitan'),
    Language('ka', 'Georgian'),
    Language('be', 'Belarusian'),
    Language('tg', 'Tajik'),
    Language('sd', 'Sindhi'),
    Language('gu', 'Gujarati'),
    Language('am', 'Amharic'),
    Language('yi', 'Yiddish'),
    Language('lo', 'Lao'),
    Language('uz', 'Uzbek'),
    Language('fo', 'Faroese'),
    Language('ht', 'Haitian creole'),
    Language('ps', 'Pashto'),
    Language('tk', 'Turkmen'),
    Language('nn', 'Nynorsk'),
    Language('mt', 'Maltese'),
    Language('sa', 'Sanskrit'),
    Language('lb', 'Luxembourgish'),
    Language('my', 'Myanmar'),
    Language('bo', 'Tibetan'),
    Language('tl', 'Tagalog'),
    Language('mg', 'Malagasy'),
    Language('as', 'Assamese'),
    Language('tt', 'Tatar'),
    Language('haw', 'Hawaiian'),
    Language('ln', 'Lingala'),
    Language('ha', 'Hausa'),
    Language('ba', 'Bashkir'),
    Language('jw', 'Javanese'),
    Language('su', 'Sundanese')
]

_TO_LANGUAGE_CODE = {
    **{language.code: language for language in LANGUAGES},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}
    
_FROM_LANGUAGE_NAME = {
    **{language.name.lower(): language for language in LANGUAGES}
}

def get_language_from_code(language_code, default=None) -> Language:
    """Return the language name from the language code."""
    return _TO_LANGUAGE_CODE.get(language_code, default)

def get_language_from_name(language, default=None) -> Language:
    """Return the language code from the language name."""
    return _FROM_LANGUAGE_NAME.get(language.lower() if language else None, default)

def get_language_names():
    """Return a list of language names."""
    return [language.name for language in LANGUAGES]

if __name__ == "__main__":
    # Test lookup
    print(get_language_from_code('en'))
    print(get_language_from_name('English'))
    
    print(get_language_names())
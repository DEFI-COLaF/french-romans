
# French 20th - 21st century - function words by grammatical nature

ARTICLES_DETERMINERS = frozenset([
    "le", "la", "l'", "les",
    "un", "une", "des",
    "du", "de la", "de l'",
    "au", "aux",
    "ce", "cet", "cette", "ces",
    "quel", "quelle", "quels", "quelles",
    "chaque",
    "tout", "toute", "tous", "toutes",
    "quelque", "quelques",
    "aucun", "aucune",
    "certain", "certaine", "certains", "certaines",
    "plusieurs",
    "autre", "autres",
])

POSSESSIVE_DETERMINERS = frozenset([
    "mon", "ma", "mes",
    "ton", "ta", "tes",
    "son", "sa", "ses",
    "notre", "nos",
    "votre", "vos",
    "leur", "leurs",
])

PERSONAL_PRONOUNS = frozenset([
    "je", "j'", "tu", "il", "elle", "on",
    "nous", "vous", "ils", "elles",
    "me", "m'", "te", "t'", "se", "s'",
    "le", "la", "l'", "les", "lui", "leur",
    "y", "en",
    "moi", "toi", "lui", "elle", "nous", "vous", "eux", "elles",
    "soi",
])

POSSESSIVE_PRONOUNS = frozenset([
    "le mien", "la mienne", "les miens", "les miennes",
    "le tien", "la tienne", "les tiens", "les tiennes",
    "le sien", "la sienne", "les siens", "les siennes",
    "le nôtre", "la nôtre", "les nôtres",
    "le vôtre", "la vôtre", "les vôtres",
    "le leur", "la leur", "les leurs",
])

DEMONSTRATIVE_PRONOUNS = frozenset([
    "ce", "cela", "ça", "ceci",
    "celui", "celle", "ceux", "celles",
    "celui-ci", "celle-ci", "ceux-ci", "celles-ci",
    "celui-là", "celle-là", "ceux-là", "celles-là",
])

INDEFINITE_PRONOUNS = frozenset([
    "chacun", "chacune", "aucun", "aucune",
    "tout", "tous", "toutes",
    "autre", "autres",
    "on",
    "quelqu'un", "quelqu’une", "quelques-uns", "quelques-unes",
    "quelque chose", "rien", "personne",
    "plusieurs",
    "nul", "nulle",
    "autrui",
])

INTERROGATIVE_PRONOUNS = frozenset([
    "qui", "que", "quoi",
    "lequel", "laquelle", "lesquels", "lesquelles",
    "auquel", "auxquels", "auxquelles",
    "duquel", "desquels", "desquelles",
    "dont",
])

PREPOSITIONS = frozenset([
    "à", "de", "dans", "en", "par", "pour", "sans", "sur", "sous",
    "chez", "avec", "entre", "contre", "vers", "selon", "parmi", "via",
    "après", "avant", "depuis", "pendant", "durant", "dès", "lors", "grâce",
    "derrière", "devant", "autour", "hors", "jusque", "jusqu'",
    "dessous", "dessus", "outre", "ès",
    "à travers", "vis-à-vis", "auprès de", "quant à", "loin de", "près de",
])

COORD_CONJUNCTIONS = frozenset(["et", "ou", "ni", "mais", "donc", "or", "car"])

SUBORD_CONJUNCTIONS = frozenset([
    "que", "si", "comme", "lorsque", "quand", "puisque",
    "parce que", "alors que", "tandis que",
    "bien que", "quoique", "afin que", "pour que", "de sorte que",
    "avant que", "après que", "dès que", "tant que", "autant que",
    "même si", "sans que",
    "à moins que", "à condition que", "pourvu que",
    "vu que", "attendu que", "dans la mesure où", "comme si",
])

ADVERBS = frozenset([
    "ici", "là", "ailleurs", "partout", "dedans", "dehors",
    "aujourd'hui", "hier", "demain", "toujours", "souvent", "parfois",
    "rarement", "jamais", "déjà", "encore", "tôt", "tard",
    "bien", "mal", "mieux", "pis",
    "très", "trop", "assez", "peu", "moins", "davantage", "autant",
    "presque", "quasiment", "simplement", "seulement", "largement",
    "à peine", "beaucoup", "tant", "tellement",
    "oui", "si", "non", "peut-être",
    "ainsi", "alors", "puis", "ensuite", "cependant", "pourtant",
    "toutefois", "d'ailleurs", "enfin", "bref", "après",
    "aussi",
    "sitôt", "volontiers", "précisément",
])

ADV_LOCUTIONS = frozenset([
    "tout de suite", "à nouveau", "de nouveau", "d'emblée", "en général",
    "en particulier", "en fait", "en effet", "en réalité", "en somme",
    "en outre", "de plus", "par contre", "au contraire", "au moins",
    "à tout le moins", "c'est-à-dire", "bien sûr", "sans doute", "d'ailleurs",
    "par ailleurs", "pour autant", "à ce propos", "à vrai dire",
    "de toute façon", "quoi qu'il en soit", "quand même", "tout de même",
    "pour l'instant", "pour le moment", "dans l'ensemble", "sur-le-champ",
    "peu à peu", "au fur et à mesure", "dorénavant", "désormais",
    "tout d'abord", "avant tout", "en premier lieu", "en dernier lieu",
    "entre-temps", "jusqu'alors", "jusqu'ici", "tout de suite après",
    "tout compte fait", "aujourd'hui même", "hier soir", "demain matin",
    "tout récemment", "tout à coup", "d'une part", "d'autre part",
    "en tout cas", "en tout temps", "en même temps", "pour ainsi dire",
    "tant bien que mal", "tout au plus",
])

NEGATIONS = frozenset([
    "ne", "n'", "n",
    "pas", "plus", "jamais", "rien", "personne",
    "guère", "aucun", "aucune",
    "nul", "nulle", "nullement", "point",
    "sans",
    "ni",
    "nulle part", "pas du tout", "plus jamais",
])

AUX_ETRE = frozenset([
    "être", "étant", "été",
    "suis", "es", "est", "sommes", "êtes", "sont",
    "étais", "était", "étions", "étiez", "étaient",
    "fus", "fut", "fûmes", "fûtes", "furent",
    "serai", "seras", "sera", "serons", "serez", "seront",
    "serais", "serait", "serions", "seriez", "seraient",
    "sois", "soit", "soyons", "soyez", "soient",
    "fusse", "fusses", "fût", "fussions", "fussiez", "fussent",
])

AUX_AVOIR = frozenset([
    "avoir", "ayant", "eu",
    "ai", "as", "a", "avons", "avez", "ont",
    "avais", "avait", "avions", "aviez", "avaient",
    "eus", "eut", "eûmes", "eûtes", "eurent",
    "aurai", "auras", "aura", "aurons", "aurez", "auront",
    "aurais", "aurait", "aurions", "auriez", "auraient",
    "aie", "aies", "ait", "ayons", "ayez", "aient",
    "eusse", "eusses", "eût", "eussions", "eussiez", "eussent",
])

# Modal verb forms (full) build:

def _forms(*cols):
    return frozenset(cols)

MODAL_VERB_FORMS = {
    "pouvoir": _forms(
        "pouvoir", "pouvant", "pu",
        "peux", "peut", "pouvons", "pouvez", "peuvent",
        "pouvais", "pouvait", "pouvions", "pouviez", "pouvaient",
        "pus", "put", "pûmes", "pûtes", "purent",
        "pourrai", "pourras", "pourra", "pourrons", "pourrez", "pourront",
        "pourrais", "pourrait", "pourrions", "pourriez", "pourraient",
        "puisse", "puisses", "puissions", "puissiez", "puissent",
        "pusse", "pusses", "pussions", "pussiez", "pussent",
    ),
    "devoir": _forms(
        "devoir", "devant", "dû",
        "dois", "doit", "devons", "devez", "doivent",
        "devais", "devait", "devions", "deviez", "devaient",
        "dus", "dut", "dûmes", "dûtes", "durent",
        "devrai", "devras", "devra", "devrons", "devrez", "devront",
        "devrais", "devrait", "devrions", "devriez", "devraient",
        "doive", "doives", "devions", "deviez", "doivent",
        "dusse", "dusses", "dussions", "dussiez", "dussent",
    ),
    "vouloir": _forms(
        "vouloir", "voulant", "voulu",
        "veux", "veut", "voulons", "voulez", "veulent",
        "voulais", "voulait", "voulions", "vouliez", "voulaient",
        "voulus", "voulu", "voulûmes", "voulûtes", "voulurent",
        "voudrai", "voudras", "voudra", "voudrons", "voudrez", "voudront",
        "voudrais", "voudrait", "voudrions", "voudriez", "voudraient",
        "veuille", "veuilles", "veuillions", "veuilliez", "veuillent",
        "voulusse", "voulusses", "voulussions", "voulussiez", "voulussent",
        "veuille", "veuillons", "veuillez",
    ),
    "savoir": _forms(
        "savoir", "sachant", "su",
        "sais", "sait", "savons", "savez", "savent",
        "savais", "savait", "savions", "saviez", "savaient",
        "sus", "sut", "sûmes", "sûtes", "surent",
        "saurai", "sauras", "saura", "saurons", "saurez", "sauront",
        "saurais", "saurait", "saurions", "sauriez", "sauraient",
        "sache", "saches", "sachions", "sachiez", "sachent",
        "susse", "susses", "sussions", "sussiez", "sussent",
        "sache", "sachons", "sachez",
    ),
    "falloir": _forms(
        "falloir", "fallu", "fallant",
        "faut",
        "fallait",
        "fallut",
        "faudra",
        "faudrait",
        "faille",
        "fallût",
    ),
    "aller": _forms(
        "aller", "allant", "allé",
        "vais", "vas", "va", "allons", "allez", "vont",
        "allais", "allait", "allions", "alliez", "allaient",
        "allai", "allas", "alla", "allâmes", "allâtes", "allèrent",
        "irai", "iras", "ira", "irons", "irez", "iront",
        "irais", "irait", "irions", "iriez", "iraient",
        "aille", "ailles", "aille", "allions", "alliez", "aillent",
        "allasse", "allasses", "allât", "allassions", "allassiez", "allassent",
        "va", "allons", "allez",
    ),
    "venir": _forms(
        "venir", "venant", "venu",
        "viens", "vient", "venons", "venez", "viennent",
        "venais", "venait", "venions", "veniez", "venaient",
        "vins", "vint", "vînmes", "vîntes", "vinrent",
        "viendrai", "viendras", "viendra", "viendrons", "viendrez", "viendront",
        "viendrais", "viendrait", "viendrions", "viendriez", "viendraient",
        "vienne", "viennes", "vienne", "venions", "veniez", "viennent",
        "vinsse", "vinsses", "vînt", "vinssions", "vinssiez", "vinssent",
        "viens", "venons", "venez",
    ),
}

MODALS_FULL = frozenset().union(*MODAL_VERB_FORMS.values())

categories = {
    "articles": sorted(ARTICLES_DETERMINERS),
    "poss_det": sorted(POSSESSIVE_DETERMINERS),
    "pers_pron": sorted(PERSONAL_PRONOUNS),
    "poss_pron": sorted(POSSESSIVE_PRONOUNS),
    "dem_pron": sorted(DEMONSTRATIVE_PRONOUNS),
    "indef_pron": sorted(INDEFINITE_PRONOUNS),
    "inter_pron": sorted(INTERROGATIVE_PRONOUNS),
    "prepositions": sorted(PREPOSITIONS),
    "coord_conj": sorted(COORD_CONJUNCTIONS),
    "subord_conj": sorted(SUBORD_CONJUNCTIONS),
    "adverbs": sorted(ADVERBS),
    "adv_locutions": sorted(ADV_LOCUTIONS),
    "negations": sorted(NEGATIONS),
    "aux_être": sorted(AUX_ETRE),
    "aux_avoir": sorted(AUX_AVOIR),
    "modals_full": sorted(MODALS_FULL),
}

data = {
    "name": "French – XXIe siècle",
    "language": "fr",
    "period": "21c",
    "categories": categories,
}

ALL = {word for wordset in categories.values() for word in wordset}
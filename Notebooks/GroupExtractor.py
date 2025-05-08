import re
import string
import numpy as np
import spacy
import pandas as pd
import joblib
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

AdultKeywords = ['porn', 'videos', 'sex', 'xhamster', 'amateur', 'gay', 'straight', 'fuck', 'teen', 'ass', 'cock', 'czech', 'anal', 'shemale', 'girl', 'tits', 'blowjob', 'milf', 'asian', 'pussy', 'premium', 'black', 'hot', 'hardcore', 'cum', 'pornstars', 'blonde', 'brunette', 'dick', 'cumshot', 'solo', 'republic', 'interracial', 'pornstar', 'porno', 'exclusive', 'mature', 'handjob', 'lesbian', 'pov', 'wife', 'webcam', 'creampie', 'masturbation', 'brazzers', 'fetish', 'ebony', 'latina', 'orgasm', 'cam', 'watch', 'gangbang', 'japanese', 'guy', 'gifs', 'threesome', 'verified', 'categories', 'fucks', 'embed', 'bbw', 'category', 'love', 'lingerie', 'english', 'toy', 'fisting', 'homemade', 'redhead', 'pornhubcom', 'babe', 'stockings', 'hentai', 'indian', 'sexy', 'movie', 'favorite', 'hard', 'bdsm', 'model', 'playlist', 'femdom', 'fucking', 'pornhub', 'discover', 'rated', 'channels', 'step', 'upload', 'outdoor', 'horny', 'huge', 'twink', 'albums', 'facial', 'massage', 'latin', 'xxx', 'reality', 'bbc']
ComputersKeywords = ['software', 'download', 'solution', 'file', 'services', 'company', 'online', 'technology', 'version', 'development', 'data', 'security', 'application', 'code', 'tool', 'copyright', 'solutions', 'windows', 'build', 'network', 'base', 'server', 'source', 'cloud', 'internet', 'mobile', 'check', 'linux', 'faq', 'database', 'mail', 'documentation', 'host', 'developer', 'technical', 'log', 'programming', 'guide', 'interface', 'standard', 'connect', 'close', 'mac', 'current', 'domain', 'hardware', 'theme', 'advanced', 'key', 'integration', 'virtual', 'automation', 'desktop', 'print', 'model', 'powerful', 'tools', 'class', 'article', 'java', 'package', 'machine', 'bug', 'display', 'api', 'engine', 'testing', 'tech', 'editor', 'environment', 'screen', 'javascript', 'suite', 'multiple', 'collection', 'field', 'size', 'communication', 'template', 'storage', 'plugin', 'multi', 'install', 'remote', 'module', 'ready', 'bit', 'directory', 'upgrade', 'tutorial', 'android', 'printer', 'script', 'implement', 'window', 'usb', 'excel', 'sql', 'xml', 'joomla']
GamesKeywords = ['game', 'play', 'games', 'online', 'download', 'player', 'forum', 'version', 'copyright', 'casino', 'create', 'club', 'check', 'gaming', 'community', 'team', 'server', 'character', 'win', 'bridge', 'chess', 'tournament', 'puzzle', 'level', 'fantasy', 'war', 'poker', 'development', 'adventure', 'friend', 'wiki', 'rules', 'rpg', 'battle', 'strategy', 'arcade', 'season', 'magic', 'virtual', 'guild', 'nintendo', 'league', 'quest', 'role', 'challenge', 'patch', 'clan', 'chat', 'dragon', 'xbox', 'discord', 'score', 'slot', 'campaign', 'combat', 'pinball', 'fight', 'mod', 'wars', 'winner', 'playing', 'mode', 'bonus', 'multiplayer', 'weapon', 'mission', 'universe', 'tournaments', 'sports', 'beta', 'major', 'playstation', 'players', 'key', 'award', 'dice', 'wow', 'dungeon', 'upgrade', 'kill', 'cheat', 'steam', 'lottery', 'sudoku', 'betting', 'blue', 'gambling', 'gameplay', 'bingo', 'warcraft', 'expansion', 'hero', 'solitaire', 'increase', 'armor', 'beat', 'sims', 'blackjack', 'minecraft', 'collector']
HealthKeywords = ['patient', 'services', 'medical', 'treatment', 'pet', 'medicine', 'therapy', 'donate', 'hospital', 'surgery', 'volunteer', 'cancer', 'clinical', 'appointment', 'child', 'disease', 'veterinary', 'emergency', 'clinic', 'pain', 'safety', 'animal', 'body', 'doctor', 'virtual', 'healthcare', 'eye', 'woman', 'check', 'donation', 'recovery', 'healing', 'journal', 'physician', 'nursing', 'heart', 'adult', 'healthy', 'disorder', 'wellness', 'vision', 'provider', 'drug', 'safe', 'treat', 'insurance', 'awareness', 'institute', 'prevention', 'breast', 'loss', 'primary', 'foot', 'physical', 'skin', 'dental', 'stress', 'pregnancy', 'acupuncture', 'vaccine', 'nurse', 'nutrition', 'cat', 'dog', 'testing', 'risk', 'injury', 'addiction', 'surgical', 'ensure', 'weight', 'symptom', 'patients', 'anxiety', 'massage', 'diagnosis', 'consultation', 'women', 'syndrome', 'procedure', 'manage', 'brain', 'blood', 'rehabilitation', 'pharmacy', 'baby', 'laser', 'specialist', 'sleep', 'pediatric', 'medication', 'chronic', 'diet', 'hair', 'vet', 'fertility', 'surgeon', 'cell', 'plastic', 'cosmetic']
NewsKeywords = ['sports', 'local', 'business', 'opinion', 'editor', 'edition', 'media', 'submit', 'online', 'obituaries', 'university', 'entertainment', 'weather', 'police', 'newspaper', 'travel', 'education', 'digital', 'game', 'events', 'culture', 'advertising', 'government', 'country', 'stories', 'editorial', 'magazine', 'advertise', 'president', 'article', 'american', 'trump', 'vaccine', 'journalism', 'district', 'company', 'council', 'archives', 'lifestyle', 'record', 'advertisement', 'baseball', 'law', 'international', 'coronavirus', 'election', 'market', 'canada', 'publish', 'politics', 'basketball', 'technology', 'result', 'listen', 'film', 'death', 'job', 'check', 'science', 'football', 'subscriber', 'current', 'legal', 'valley', 'industry', 'sell', 'access', 'awards', 'journalist', 'build', 'court', 'war', 'subscription', 'department', 'network', 'jobs', 'reporter', 'vote', 'road', 'crime', 'charge', 'lake', 'nation', 'age', 'hit', 'feel', 'development', 'resident', 'friend', 'river', 'grow', 'leader', 'official', 'person', 'car', 'light', 'bank', 'load', 'hotel', 'usd']
RecreationKeywords = ['club', 'wine', 'dog', 'book', 'family', 'car', 'breed', 'travel', 'calendar', 'reserve', 'trip', 'guide', 'fishing', 'life', 'fun', 'love', 'map', 'special', 'training', 'road', 'river', 'tour', 'fly', 'activity', 'hunt', 'team', 'puppy', 'water', 'volunteer', 'boat', 'country', 'lake', 'adventure', 'stay', 'rescue', 'collection', 'food', 'friend', 'beer', 'field', 'wines', 'cat', 'hunting', 'bird', 'winery', 'canada', 'safety', 'dive', 'camping', 'mountain', 'ride', 'sailing', 'explore', 'fish', 'valley', 'air', 'drive', 'guest', 'beautiful', 'house', 'school', 'pet', 'yacht', 'trail', 'night', 'diving', 'australia', 'vineyard', 'california', 'equipment', 'flight', 'season', 'weekend', 'cover', 'resort', 'charter', 'spring', 'weather', 'cruise', 'sport', 'win', 'holiday', 'outdoor', 'discover', 'tours', 'race', 'region', 'tasting', 'animal', 'game', 'bear', 'wildlife', 'walk', 'sea', 'lodge', 'scuba', 'islands', 'hotel', 'destination', 'vacation']
ReferenceKeywords = ['student', 'university', 'college', 'campus', 'education', 'schedule', 'faculty', 'programs', 'resources', 'library', 'graduate', 'students', 'study', 'learning', 'science', 'academic', 'alumni', 'book', 'class', 'collection', 'department', 'technology', 'undergraduate', 'admission', 'mission', 'studies', 'engineering', 'map', 'institute', 'degree', 'courses', 'teacher', 'exhibit', 'application', 'sciences', 'educational', 'global', 'academics', 'teaching', 'scholarship', 'alumnus', 'collections', 'master', 'registration', 'curriculum', 'hours', 'tuition', 'exhibition', 'professor', 'publications', 'scholarships', 'phd', 'image', 'facilities', 'word', 'teach', 'cultural', 'workshop', 'activities', 'archives', 'grant', 'classroom', 'programme', 'grow', 'catalog', 'academy', 'grade', 'maps', 'honor', 'classes', 'woman', 'institution', 'graduation', 'libraries', 'technical', 'journal', 'degrees', 'colleges', 'housing', 'accreditation', 'exam', 'commencement', 'prospective', 'equity', 'semester', 'athlete', 'postgraduate', 'dean', 'departments', 'math', 'mba', 'ncaa', 'biology', 'requirements', 'bachelor', 'physics', 'honors', 'diploma', 'minor', 'psychology']
ScienceKeywords = ['science', 'resources', 'development', 'technology', 'analysis', 'water', 'data', 'study', 'life', 'environmental', 'engineering', 'journal', 'energy', 'scientific', 'testing', 'model', 'material', 'laboratory', 'technical', 'safety', 'institute', 'space', 'human', 'land', 'global', 'lab', 'plant', 'air', 'cell', 'environment', 'conservation', 'food', 'industrial', 'animal', 'method', 'sciences', 'natural', 'impact', 'gas', 'earth', 'registration', 'phd', 'box', 'production', 'physics', 'medical', 'theory', 'manufacturing', 'astronomy', 'volunteer', 'climate', 'chemical', 'function', 'scientist', 'nature', 'biology', 'measurement', 'watch', 'engineer', 'math', 'platform', 'processing', 'marine', 'weather', 'chemistry', 'record', 'australia', 'leadership', 'researcher', 'solar', 'effective', 'county', 'grant', 'specie', 'child', 'category', 'webinar', 'temperature', 'clinical', 'green', 'telescope', 'measure', 'sustainable', 'imaging', 'position', 'wildlife', 'launch', 'waste', 'protein', 'disease', 'dna', 'ocean', 'expand', 'force', 'soil', 'filter', 'molecular', 'organic', 'carbon', 'gene']
ShoppingKeywords = ['products', 'sale', 'shipping', 'store', 'stock', 'purchase', 'delivery', 'shopping', 'ship', 'catalog', 'services', 'supply', 'items', 'wholesale', 'checkout', 'category', 'craft', 'pack', 'tool', 'supplies', 'payment', 'produce', 'returns', 'plant', 'categories', 'discount', 'vintage', 'table', 'gear', 'limited', 'reviews', 'metal', 'basket', 'options', 'pet', 'brands', 'organic', 'wishlist', 'clothing', 'designer', 'faqs', 'fabric', 'leather', 'sets', 'sport', 'sports', 'kitchen', 'kids', 'package', 'manufacturer', 'apparel', 'beauty', 'shirt', 'shirts', 'bottle', 'sellers', 'cotton', 'toys', 'shoes', 'diamond', 'dress', 'chairs', 'baskets', 'boots', 'shoe', 'necklaces', 'plants', 'coat', 'dresses', 'womens']
SocietyKeywords = ['church', 'worship', 'ministry', 'law', 'sunday', 'god', 'attorney', 'prayer', 'sermon', 'bible', 'parish', 'christ', 'methodist', 'school', 'community', 'jesus', 'firm', 'baptist', 'lawyer', 'legal', 'injury', 'catholic', 'pastor', 'faith', 'accident', 'service', 'christian', 'child', 'youth', 'client', 'funeral', 'yoga', 'mass', 'holy', 'mission', 'family', 'camp', 'presbyterian', 'lutheran', 'litigation', 'donate', 'event', 'united', 'love', 'lodge', 'life', 'online', 'spiritual', 'temple', 'student', 'jewish', 'estate', 'congregation', 'volunteer', 'woman', 'animal', 'jun', 'resource', 'practice', 'retreat', 'bulletin', 'bankruptcy', 'live', 'meeting', 'preschool', 'calendar', 'criminal', 'adult', 'county', 'personal', 'zoom', 'shabbat', 'book', 'study', 'justice', 'court', 'business', 'class', 'chapter', 'meditation', 'post', 'serve', 'divorce', 'patent', 'fellowship', 'gospel', 'association', 'connect', 'music', 'society', 'care', 'unitarian', 'international', 'wedding', 'district', 'saint', 'israel', 'episcopal', 'bishop', 'membership']
SportsKeywords = ['golf', 'club', 'horse', 'league', 'race', 'soccer', 'ski', 'football', 'team', 'ride', 'tee', 'coach', 'player', 'bike', 'shot', 'martial', 'hockey', 'rugby', 'aikido', 'season', 'karate', 'ticket', 'class', 'junior', 'game', 'paintball', 'tournament', 'camp', 'play', 'event', 'trail', 'youth', 'hole', 'training', 'news', 'cycling', 'lesson', 'cup', 'academy', 'association', 'sport', 'schedule', 'stallion', 'championship', 'mountain', 'racing', 'sponsor', 'referee', 'tour', 'park', 'farm', 'skate', 'fixture', 'post', 'match', 'art', 'swimming', 'book', 'instructor', 'dojo', 'bicycle', 'country', 'tennis', 'fencing', 'marathon', 'registration', 'track', 'skating', 'surf', 'dressage', 'clubhouse', 'rider', 'swim', 'photo', 'bowling', 'champion', 'field', 'runner', 'run', 'resort', 'sale', 'goal', 'competition', 'trip', 'standing', 'forum', 'breed', 'cricket', 'canoe', 'ranch', 'winger', 'school', 'division', 'mare', 'summer', 'kayak', 'valley', 'save', 'river', 'bowl']

html_scaler = joblib.load('html_scaler.joblib')
lda = joblib.load('lda.joblib')
LDACountvectorizer = joblib.load('LDACountvectorizer.joblib')
meta_title_scaler = joblib.load('meta_title_scaler.joblib')
percentagesScaler = joblib.load('percentagesScaler.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')


with open("stopwords-en.txt", "r") as file:
    stopwords_list = file.read().splitlines()

stopwords = set(stopwords_list)
custom_stopwords = set([
    # First batch of words with no meaning
    'contact', 'service', 'policy', 'site', 'privacy', 'support', 'email', 'blog',
    'post', 'learn', 'read', 'offer', 'provide', 'include', 'click', 'update',
    'feature', 'link', 'search', 'website', 'program', 'start', 'view', 'resource',
    'experience', 'list', 'free', 'info', 'shop', 'video', 'share', 'member',
    'add', 'start', 'work', 'order', 'day', 'people', 'history', 'office',
    'time', 'year', 'event', 'national', 'state', 'high', 'month', 'week', 'open',
    'cookies', 'menu', 'cart', 'browser', 'select', 'choose', 'hope', 'enjoy',

    # Social media/web
    'facebook', 'twitter', 'youtube', 'instagram', 'account', 'cookie', 'subscribe',
    'newsletter', 'sign', 'message', 'comment', 'form', 'login', 'user', 'member',
    'join', 'write', 'update', 'search', 'review',

    # Dates
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december', 'year', 'today', 'yesterday', 'tomorrow', 'datum', 'date',
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',

    # Days of the week
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',

    # Places with no meaning
    'regional', 'albuquerque', 'chicago', 'minneapolis', 'philadelphia', 'phoenix', 'rhode', 'island', 'scottsdale', 'washington', 'wisconsin', 'michigan',
    'bay', 'beach', 'dakota', 'florida', 'georgia', 'hampshire', 'harbor', 'iowa', 'maine',  'missouri', 'park', 'virginia', 'vista', 'wisconsin', 'massachusetts',
    'minnesota',

    # Wrds holding no weight or too general to hold any
    'skip', 'content', 'main', 'term', 'condition', 'toggle', 'navigation', 'wordpress', 'social', 'medium', 'upcoming', 'event',
    'photo', 'gallery', 'news', 'frequently', 'question', 'ask', 'press', 'release', 'quick', 'link', 'continue', 'read', 'phone', 'fax', 'answer', 'question',
    'board', 'director', 'real', 'estate', 'los', 'angeles', 'new', 'york', 'city', 'san', 'francisco', 'power', 'united', 'kingdom', 'states', 'america', 'fran', 'ais',
    'north', 'carolina', 'las', 'vegas', 'annual', 'report', 'highly', 'recommend', 'rss', 'feed', 'white', 'paper', 'hong', 'kong', 'credit', 'card', 'mental', 'health', 'public', 'save', 'money',
    'annual', 'meeting', 'wide', 'range', 'care', 'gift', 'professional', 'live', 'stream', 'quality', 'product', 'project', 'management', 'meet', 'nonprofit', 'organization', 'blogthis', 'pinter',
    'design', 'success', 'story', 'summer', 'camp', 'chain', 'register', 'trademark', 'username', 'password', 'certificate', 'plan', 'visit', 'regular', 'price', 'covid', 'pandemic', 'south', 'africa', 'west', 'east', 'regional',
])
stopwords.update(custom_stopwords)
stopwords = sorted(stopwords)

def lemmatize_text(nlp, text):
    # Process the text through the spaCy NLP pipeline
    doc = nlp(text)
    # Return the lemmatized text
    return " ".join([token.lemma_ for token in doc])

def keyword_percentage(text, keywords):
    words = text.split()
    keyword_count = sum(1 for word in words if word in keywords)
    return (keyword_count / len(keywords)) * 100 if words else 0

def extract_text_features(text):
    words = text.split()
    word_count = len(words)
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    unique_word_count = len(set(words))
    return word_count, avg_word_length, unique_word_count

def further_clean_text(text, stopwords):
    # Normalize spaces; replaces all kinds of whitespace with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove all numbers (digits) from the text
    text = re.sub(r'\d+', '', text)

    # Remove non-English characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Convert text to lower case to standardize for stopwords removal
    text = text.lower()

    # Split text into words, remove short words and stopwords
    text = ' '.join([word for word in text.split() if len(word) >= 3 and word not in stopwords])
    text = text.strip()

    return text

def ExtractFeatures(html_list):
    # Load spaCy and scalers
    spacy.require_gpu()
    nlp = spacy.load('en_core_web_md')
    nlp.max_length = 4000000

    # Step 1: Parse HTMLs
    soup_list = [BeautifulSoup(html, 'lxml') for html in html_list]

    def extract_basic_info(soup):
        text_content = soup.get_text(separator=" ")
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        title = soup.find('title')
        return {
            'TextContent': text_content,
            'meta_description': meta_desc['content'] if meta_desc and 'content' in meta_desc.attrs else '',
            'title': title.get_text() if title else ''
        }
    
    text_info_list = [extract_basic_info(soup) for soup in soup_list]
    df = pd.DataFrame(text_info_list)
    df['RawHTML'] = html_list

    # Step 2: Text preprocessing
    df['meta_description'] = [lemmatize_text(nlp, further_clean_text(t, stopwords)) for t in df['meta_description']]
    df['title'] = [lemmatize_text(nlp, further_clean_text(t, stopwords)) for t in df['title']]
    df['TextContent'] = [lemmatize_text(nlp, further_clean_text(t, stopwords)) for t in df['TextContent']]

    # Step 3: Keyword percentages for meta and title
    keyword_percent_cols = []
    for column in ['meta_description', 'title']:
        for cat_name in ['Adult', 'Computers', 'Games', 'Health', 'News', 'Recreation', 'Reference', 'Science', 'Shopping', 'Society', 'Sports']:
            col_name = f"{column}_{cat_name}KeywordPercentage"
            df[col_name] = df[column].apply(lambda x: keyword_percentage(x, globals()[f'{cat_name}Keywords']))
            keyword_percent_cols.append(col_name)

    meta_title_keyword_features = pd.DataFrame(meta_title_scaler.transform(df[keyword_percent_cols]), columns=keyword_percent_cols)

    # Step 4: HTML Tag-based features
    def extract_tag_features(soup, text_len, html_len):
        tags = [tag.name for tag in soup.find_all()]
        return {
            'unique_html_tags': len(set(tags)),
            'h1_count': tags.count('h1'),
            'h2_count': tags.count('h2'),
            'h3_count': tags.count('h3'),
            'paragraph_count': tags.count('p'),
            'text_to_html_ratio': text_len / html_len if html_len else 0,
            'script_count': tags.count('script'),
            'img_count': tags.count('img'),
            'form_count': tags.count('form'),
            'anchor_count': tags.count('a'),
            'iframe_count': tags.count('iframe'),
            'article_count': tags.count('article'),
            'button_count': tags.count('button'),
            'img_to_text_ratio': tags.count('img') / text_len if text_len else 0,
            'meta_desc_len': len(soup.find('meta', attrs={'name': 'description'})['content']) if soup.find('meta', attrs={'name': 'description'}) and 'content' in soup.find('meta', attrs={'name': 'description'}).attrs else 0,
            'link_density': sum(len(tag.get_text()) for tag in soup.find_all('a')) / text_len if text_len else 0,
            'video_count': tags.count('video'),
            'audio_count': tags.count('audio'),
            'section_count': tags.count('section'),
            'div_count': tags.count('div'),
            'ul_count': tags.count('ul'),
            'ol_count': tags.count('ol'),
            'stylesheet_count': len(soup.find_all('link', {'rel': 'stylesheet'})),
            'external_script_count': len(soup.find_all('script', {'src': True})),
            'input_count': tags.count('input'),
            'select_count': tags.count('select'),
            'textarea_count': tags.count('textarea'),
            'social_media_links_count': sum(1 for tag in soup.find_all('a', href=True) if any(social in tag['href'] for social in ['facebook.com', 'twitter.com', 'instagram.com']))
        }

    tag_features = [extract_tag_features(soup, len(text), len(html)) for soup, text, html in zip(soup_list, df['TextContent'], df['RawHTML'])]
    tag_df = pd.DataFrame(tag_features)

    # Step 5: Word stats (word count, avg word length, unique words)
    word_stats = df['TextContent'].apply(extract_text_features)
    word_stats_df = pd.DataFrame(word_stats.tolist(), columns=['word_count', 'avg_word_length', 'unique_word_count'])

    combined_html_text_features_input = pd.concat([tag_df, word_stats_df], axis=1)
    html_text_features_scaled = html_scaler.transform(combined_html_text_features_input)
    html_text_features = pd.DataFrame(html_text_features_scaled, columns=combined_html_text_features_input.columns)

    # Step 6: Topic modeling with LDA
    X_counts = LDACountvectorizer.transform(df['TextContent'])
    topic_features = pd.DataFrame(lda.transform(X_counts), columns=[f"Topic_{i}" for i in range(16)])

    # Step 7: Keyword percentages from full text
    text_keyword_cols = []
    for cat_name in ['Adult', 'Computers', 'Games', 'Health', 'News', 'Recreation', 'Reference', 'Science', 'Shopping', 'Society', 'Sports']:
        col_name = f"{cat_name}KeywordPercentage"
        df[col_name] = df['TextContent'].apply(lambda x: keyword_percentage(x, globals()[f'{cat_name}Keywords']))
        text_keyword_cols.append(col_name)

    text_keyword_features = pd.DataFrame(percentagesScaler.transform(df[text_keyword_cols]), columns=text_keyword_cols)

    tfidf_matrix = tfidf_vectorizer.transform(df['TextContent'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Step 9: Final Concatenation
    combined_df = pd.concat([
        tfidf_df.reset_index(drop=True),
        html_text_features.reset_index(drop=True),
        text_keyword_features.reset_index(drop=True),
        topic_features.reset_index(drop=True),
        meta_title_keyword_features.reset_index(drop=True)
    ], axis=1)

    if 'category' in combined_df.columns:
        combined_df.drop(columns='category', inplace=True)


    return combined_df






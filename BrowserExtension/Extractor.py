import re
import string
import numpy as np
import joblib
import spacy
import pandas as pd
from bs4 import BeautifulSoup

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

html_scaler = joblib.load('scalers/html_scaler.joblib')
lda = joblib.load('scalers/lda.joblib')
LDACountvectorizer = joblib.load('scalers/LDACountvectorizer.joblib')
meta_title_scaler = joblib.load('scalers/meta_title_scaler.joblib')
percentagesScaler = joblib.load('scalers/percentagesScaler.joblib')
tfidf_vectorizer = joblib.load('scalers/tfidf_vectorizer.joblib')

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

    # Words holding no weight or too general to hold any
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

def ExtractFeatures(html):
    
    # Initialize NLP pipeliner
    spacy.require_gpu()
    nlp = spacy.load('en_core_web_md')
    nlp.max_length = 4000000
    
    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html, 'lxml')
    

    # Fetch meta and title contents
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    title = soup.find('title')
    meta_desc_content = meta_desc['content'] if meta_desc and 'content' in meta_desc.attrs else ""
    title_content = title.get_text() if title else ""
    
    # Get text and strip leading/trailing whitespace
    text_content = soup.get_text(separator=" ")

    # Append new row to the DataFrame
    new_data = {
        'RawHTML': [html],
        'TextContent': [text_content],
        'meta_description': [meta_desc_content],
        'title': [title_content]
    }

    df = pd.DataFrame(new_data)

    df['meta_description'] = df['meta_description'].apply(lambda x: lemmatize_text(nlp, x))
    df['title'] = df['title'].apply(lambda x: lemmatize_text(nlp, x))

    # Calculate keyword percentages for each category and add them to the DataFrame
    for column in ['meta_description', 'title']:
        df[f'{column}_AdultKeywordPercentage'] = df[column].apply(lambda x: keyword_percentage(x, AdultKeywords))
        df[f'{column}_ComputersKeywordPercentage'] = df[column].apply(lambda x: keyword_percentage(x, ComputersKeywords))
        df[f'{column}_GamesKeywordPercentage'] = df[column].apply(lambda x: keyword_percentage(x, GamesKeywords))
        df[f'{column}_HealthKeywordPercentage'] = df[column].apply(lambda x: keyword_percentage(x, HealthKeywords))
        df[f'{column}_NewsKeywordPercentage'] = df[column].apply(lambda x: keyword_percentage(x, NewsKeywords))
        df[f'{column}_RecreationKeywordPercentage'] = df[column].apply(lambda x: keyword_percentage(x, RecreationKeywords))
        df[f'{column}_ReferenceKeywordPercentage'] = df[column].apply(lambda x: keyword_percentage(x, ReferenceKeywords))
        df[f'{column}_ScienceKeywordPercentage'] = df[column].apply(lambda x: keyword_percentage(x, ScienceKeywords))
        df[f'{column}_ShoppingKeywordPercentage'] = df[column].apply(lambda x: keyword_percentage(x, ShoppingKeywords))
        df[f'{column}_SocietyKeywordPercentage'] = df[column].apply(lambda x: keyword_percentage(x, SocietyKeywords))
        df[f'{column}_SportsKeywordPercentage'] = df[column].apply(lambda x: keyword_percentage(x, SportsKeywords))
    
    # Select the keyword percentage columns for meta description and title
    meta_title_keyword_columns = [f'{column}_{category}KeywordPercentage' for column in ['meta_description', 'title'] for category in ['Adult', 'Computers', 'Games', 'Health', 'News', 'Recreation', 'Reference', 'Science', 'Shopping', 'Society', 'Sports']]

    # Scale the keyword percentage features
    meta_title_keyword_features = meta_title_scaler.transform(df[meta_title_keyword_columns])
    meta_title_keyword_features = pd.DataFrame(meta_title_keyword_features, columns=meta_title_keyword_columns)


    tags = [tag.name for tag in soup.find_all()]
    df['unique_html_tags'] = len(set(tags))
    df['h1_count'] = tags.count('h1')
    df['h2_count'] = tags.count('h2')
    df['h3_count'] = tags.count('h3')
    df['paragraph_count'] = tags.count('p')
    df['text_to_html_ratio'] = len(text_content) / len(html) if len(html) > 0 else 0
    df['script_count'] = tags.count('script')
    df['img_count'] = tags.count('img')
    df['form_count'] = tags.count('form')
    df['anchor_count'] = tags.count('a')
    df['iframe_count'] = tags.count('iframe')
    df['article_count'] = tags.count('article')
    df['button_count'] = tags.count('button')

    df['img_to_text_ratio'] = tags.count('img') / len(text_content) if len(text_content) > 0 else 0
    df['meta_desc_len'] = len(meta_desc['content']) if meta_desc and 'content' in meta_desc.attrs else 0

    link_text = sum(len(tag.get_text()) for tag in soup.find_all('a'))
    total_text = len(text_content)
    df['link_density'] = link_text / total_text if total_text > 0 else 0

    df['video_count'] = tags.count('video')
    df['audio_count'] = tags.count('audio')

    df['section_count'] = tags.count('section')
    df['div_count'] = tags.count('div')
    df['nav_count'] = tags.count('nav')
    df['footer_count'] = tags.count('footer')

    df['ul_count'] = tags.count('ul')
    df['ol_count'] = tags.count('ol')

    df['stylesheet_count'] = len(soup.find_all('link', {'rel': 'stylesheet'}))
    df['external_script_count'] = len(soup.find_all('script', {'src': True}))

    df['input_count'] = tags.count('input')
    df['select_count'] = tags.count('select')
    df['textarea_count'] = tags.count('textarea')
    df['social_media_links_count'] = sum(1 for tag in soup.find_all('a', href=True) if "facebook.com" in tag['href'] or "twitter.com" in tag['href'] or "instagram.com" in tag['href'])

    df[['word_count', 'avg_word_length', 'unique_word_count']] = df['TextContent'].apply(lambda x: pd.Series(extract_text_features(x)))

    html_text_feature_columns = ['unique_html_tags', 'h1_count', 'h2_count', 'h3_count', 'paragraph_count', 'text_to_html_ratio', 
                             'script_count', 'img_count', 'form_count', 'anchor_count', 'iframe_count', 'article_count', 
                             'button_count', 'img_to_text_ratio', 'meta_desc_len', 'link_density', 'video_count', 'audio_count', 'section_count', 'div_count', 'ul_count', 'ol_count', 'stylesheet_count', 'external_script_count',
                             'input_count', 'select_count', 'textarea_count', 'social_media_links_count',
                             'word_count', 'avg_word_length', 'unique_word_count']
    
    html_text_features = html_scaler.transform(df[html_text_feature_columns])
    html_text_features = pd.DataFrame(html_text_features, columns=html_text_feature_columns)
    meta_title_keyword_features = pd.DataFrame(meta_title_keyword_features, columns=meta_title_keyword_columns)

  
    df['TextContent'] = df['TextContent'].apply(lambda x: further_clean_text(x, stopwords))
    df['TextContent'] = df['TextContent'].apply(lambda x: lemmatize_text(nlp, x))

    # LDA part of the extraction
    X_counts = LDACountvectorizer.transform(df['TextContent'])
    X_topics = lda.transform(X_counts)

    topic_features = pd.DataFrame(X_topics, columns=[f'Topic_{i}' for i in range(16)])

    df['AdultKeywordPercentage'] = df['TextContent'].apply(lambda x: keyword_percentage(x, AdultKeywords))
    df['ComputersKeywordPercentage'] = df['TextContent'].apply(lambda x: keyword_percentage(x, ComputersKeywords))
    df['GamesKeywordPercentage'] = df['TextContent'].apply(lambda x: keyword_percentage(x, GamesKeywords))
    df['HealthKeywordPercentage'] = df['TextContent'].apply(lambda x: keyword_percentage(x, HealthKeywords))
    df['NewsKeywordPercentage'] = df['TextContent'].apply(lambda x: keyword_percentage(x, NewsKeywords))
    df['RecreationKeywordPercentage'] = df['TextContent'].apply(lambda x: keyword_percentage(x, RecreationKeywords))
    df['ReferenceKeywordPercentage'] = df['TextContent'].apply(lambda x: keyword_percentage(x, ReferenceKeywords))
    df['ScienceKeywordPercentage'] = df['TextContent'].apply(lambda x: keyword_percentage(x, ScienceKeywords))
    df['ShoppingKeywordPercentage'] = df['TextContent'].apply(lambda x: keyword_percentage(x, ShoppingKeywords))
    df['SocietyKeywordPercentage'] = df['TextContent'].apply(lambda x: keyword_percentage(x, SocietyKeywords))
    df['SportsKeywordPercentage'] = df['TextContent'].apply(lambda x: keyword_percentage(x, SportsKeywords))

    # Select the keyword percentage columns
    keyword_columns = ['AdultKeywordPercentage', 'ComputersKeywordPercentage', 
                    'GamesKeywordPercentage', 'HealthKeywordPercentage',
                    'NewsKeywordPercentage', 'RecreationKeywordPercentage', 'ReferenceKeywordPercentage', 
                    'ScienceKeywordPercentage', 'ShoppingKeywordPercentage', 'SocietyKeywordPercentage', 
                    'SportsKeywordPercentage']

    # Scale the keyword percentage features
    keyword_features = percentagesScaler.transform(df[keyword_columns])
    keyword_features = pd.DataFrame(keyword_features, columns=keyword_columns)

    # Fit and transform the sample
    tfidf_matrix_final = tfidf_vectorizer.transform(df['TextContent'])

    tfidf_df = pd.DataFrame(tfidf_matrix_final.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_df.drop('category', axis=1, inplace=True)

    # Concatenate the HTML features, keyword features, topic features, and bigram features with the selected features
    combined_features = np.hstack((tfidf_df.values, html_text_features.values, keyword_features.values, topic_features.values, meta_title_keyword_features.values))

    combined_feature_names = list(tfidf_df.columns) + html_text_feature_columns + keyword_columns + topic_features.columns.tolist() + meta_title_keyword_columns
    combined_df = pd.DataFrame(combined_features, columns=combined_feature_names)

    return combined_df






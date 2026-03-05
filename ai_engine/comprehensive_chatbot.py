"""Comprehensive Chatbot with Access to ALL Datasets + Multi-AI Support."""
import datetime as _dt
import pandas as pd
from typing import Dict, List, Optional
import os
import json
from collections import deque
from ai_engine.unified_dataset_loader import UnifiedDatasetLoader
from ai_engine.gemini_integration import GeminiNutritionAI
from ai_engine.langchain_ai import get_langchain_ai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()


class ComprehensiveChatbot:
    """Chatbot that can answer from all available datasets."""
    
    # Quality thresholds for answer evaluation
    MIN_QUALITY_ANSWER_LENGTH = 100  # Minimum characters for a quality answer
    
    def __init__(self):
        """Initialize chatbot with unified dataset loader and multiple AI providers."""
        self.unified_loader = UnifiedDatasetLoader()
        self.datasets = {}
        self.knowledge_base = {}
        
        # AI model timeout (default 2.5 seconds to ensure <3s total response time)
        self.ai_timeout = float(os.getenv('AI_TIMEOUT_SECONDS', '2.5'))

        # Request limiting - More permissive default (20/min instead of 10/min)
        self.rate_limit_per_min = int(os.getenv('CHATBOT_RATE_LIMIT_PER_MIN', '20'))
        self._recent_calls = deque()
        
        # Response caching for frequently asked questions
        self._response_cache = {}
        self._cache_ttl = int(os.getenv('CACHE_TTL_SECONDS', '3600'))  # 1 hour default
        
        # Initialize AI providers with fallback chain
        # 1. Gemini AI (Google's model - fast and good)
        self.gemini_ai = GeminiNutritionAI()
        
        # 2. LangChain + HuggingFace (fallback with multiple options)
        self.langchain_ai = get_langchain_ai()
        
        # 3. BERT + Flan-T5 (local AI models for semantic search and generation)
        try:
            from ai_engine.bert_flan_engine import get_engine
            self.bert_flan_engine = get_engine()
        except Exception as e:
            print(f"⚠ BERT+Flan-T5 engine not available: {e}")
            self.bert_flan_engine = None
        
        # Load all datasets through unified loader
        self._load_all_datasets()
        
        # Build comprehensive knowledge base
        self._build_comprehensive_knowledge_base()
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE PREGNANCY CHATBOT INITIALIZED")
        print(f"{'='*80}")
        print(f"✓ Loaded {len(self.datasets)} dataset categories")
        print(f"✓ Total knowledge entries: {self._count_total_entries()}")
        print(f"✓ Using UnifiedDatasetLoader with {len(self.unified_loader.meals)} total meals")
        print(f"✓ Response caching enabled (TTL: {self._cache_ttl}s)")
        print(f"✓ AI timeout: {self.ai_timeout}s")
        print(f"✓ Gemini AI available: {self.gemini_ai.available}")
        print(f"✓ LangChain AI available: {self.langchain_ai.available}")
        print(f"✓ BERT+Flan-T5 available: {self.bert_flan_engine is not None}")
        if self.bert_flan_engine:
            print(f"  - BERT+Flan-T5 loading: {self.bert_flan_engine.is_loading}")
            print(f"  - BERT+Flan-T5 ready: {self.bert_flan_engine.is_ready}")
        print(f"{'='*80}\n")

    def _is_poor_quality_answer(self, answer: str) -> bool:
        """
        Check if an answer is poor quality and should trigger fallback.
        
        Args:
            answer: The answer text to evaluate
            
        Returns:
            True if answer is poor quality, False otherwise
        """
        if not answer or len(answer.strip()) < self.MIN_QUALITY_ANSWER_LENGTH:
            return True
        
        # Check for generic placeholder responses
        if 'For specific safety information' in answer:
            return True
        
        # Check for repeated generic content (indicates keyword matching failures)
        if answer.count('**General Safety Guidelines:**') > 2:
            return True
        
        return False

    def _rate_limit_allows(self) -> bool:
        """Simple per-process rate limit: defaults to 10 requests/minute."""
        now = time.time()
        window = 60
        while self._recent_calls and now - self._recent_calls[0] > window:
            self._recent_calls.popleft()
        if len(self._recent_calls) >= self.rate_limit_per_min:
            return False
        self._recent_calls.append(now)
        return True
    
    def _create_default_context(self, trimester=None, region=None, diet_type='general'):
        """
        Create a default context dictionary for Gemini AI calls.
        
        Args:
            trimester (int, optional): Pregnancy trimester (1, 2, or 3)
            region (str, optional): User's region preference (e.g., 'North', 'South')
            diet_type (str): Diet type preference (default: 'general')
        
        Returns:
            dict: Context dictionary with trimester, region, and diet_type keys
        """
        return {
            'trimester': trimester,
            'region': region,
            'diet_type': diet_type
        }
    
    def _format_ai_response(self, answer, backend: str = 'ai_model'):
        """
        Format AI-powered response with consistent styling and disclaimer.
        
        Args:
            answer (str): The AI-generated response text
        
            backend (str): The AI backend used (bert_flan_t5, gemini, langchain, ai_model)
        Returns:
            str: Formatted response with header and medical disclaimer
        """
        backend_label = {
            'bert_flan_t5': 'BERT+Flan-T5',
            'ai_model': 'AI Model',
            'gemini': 'Gemini AI',
            'langchain': 'LangChain AI'
        }.get(backend, 'AI Model')
        
        return f"🤖 **AI-Powered Answer ({backend_label}):**\n\n{answer}\n\n💡 Note: This is AI-generated advice. Always consult your doctor for personalized guidance."
    
    def _load_all_datasets(self):
        """Load all available datasets from data folder."""
        base_path = os.path.join(os.path.dirname(__file__), '../data')
        
        # Data folder 1: Regional meal plans
        self.datasets['regional_meals'] = {
            'northveg': self._safe_load_csv(os.path.join(base_path, 'data_1/northveg_cleaned.csv')),
            'northnonveg': self._safe_load_csv(os.path.join(base_path, 'data_1/northnonveg_cleaned.csv')),
            'southveg': self._safe_load_csv(os.path.join(base_path, 'data_1/southveg_cleaned.csv')),
            'southnonveg': self._safe_load_csv(os.path.join(base_path, 'data_1/southnonveg_cleaned.csv'))
        }
        
        # Data folder 2: Trimester-wise plans
        self.datasets['trimester_plans'] = {
            'trimester_wise': self._safe_load_csv(os.path.join(base_path, 'data_2/Trimester_Wise_Diet_Plan.csv')),
            'pregnancy_diet_trimesters': self._safe_load_csv(os.path.join(base_path, 'data_2/pregnancy_diet_1st_2nd_3rd_trimester.xlsx.csv'))
        }
        
        # Data folder 3: Seasonal diets
        self.datasets['seasonal'] = {
            'summer': self._safe_load_csv(os.path.join(base_path, 'data_3/summer_pregnancy_diet.csv')),
            'winter': self._safe_load_csv(os.path.join(base_path, 'data_3/Winter_Pregnancy_Diet.csv')),
            'monsoon': self._safe_load_csv(os.path.join(base_path, 'data_3/monsoon_diet_pregnant_women.csv'))
        }
        
        # Remaining datasets
        self.datasets['special_conditions'] = {
            'foods_to_avoid': self._safe_load_csv(os.path.join(base_path, 'remainingdatasets/foods_to_avoid_during_pregnancy_dataset.csv')),
            'diabetes_diet': self._safe_load_csv(os.path.join(base_path, 'remainingdatasets/indian_diet_diabetes_pregnancy_dataset.csv')),
            'dos_donts': self._safe_load_csv(os.path.join(base_path, 'remainingdatasets/pregnancy_dos_donts_dataset.csv')),
            'pregnancy_diet_clean': self._safe_load_csv(os.path.join(base_path, 'remainingdatasets/pregnancy_diet_clean_dataset.csv')),
            'postnatal_diet': self._safe_load_csv(os.path.join(base_path, 'remainingdatasets/postnatal_diet_india_dataset.csv')),
            'postpartum_diet': self._safe_load_csv(os.path.join(base_path, 'remainingdatasets/postpartum_diet7_structured_dataset.csv'))
        }
    
    def _safe_load_csv(self, filepath: str) -> pd.DataFrame:
        """Safely load CSV file with multiple encoding attempts."""
        try:
            if os.path.exists(filepath):
                # Try different encodings
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding)
                        print(f"✓ Loaded: {os.path.basename(filepath)} ({len(df)} rows) [encoding: {encoding}]")
                        return df
                    except UnicodeDecodeError:
                        continue
                print(f"✗ Failed to load {os.path.basename(filepath)}: Could not decode with any encoding")
        except Exception as e:
            print(f"✗ Failed to load {os.path.basename(filepath)}: {e}")
        return pd.DataFrame()
    
    def _count_total_entries(self) -> int:
        """Count total entries across all datasets."""
        total = 0
        for category in self.datasets.values():
            for df in category.values():
                total += len(df)
        return total
    
    def _build_comprehensive_knowledge_base(self):
        """Build comprehensive knowledge base from all datasets."""
        self.knowledge_base = {
            'foods_to_eat': {},
            'foods_to_avoid': {},
            'meal_plans': {},
            'trimester_specific': {1: [], 2: [], 3: []},
            'seasonal_foods': {},
            'regional_foods': {},
            'dos_donts': {'dos': [], 'donts': []},
            'special_conditions': {}
        }
        
        # Process foods to avoid
        if not self.datasets['special_conditions']['foods_to_avoid'].empty:
            df = self.datasets['special_conditions']['foods_to_avoid']
            for _, row in df.iterrows():
                food_item = row.get('Food_Item', '')
                if food_item:
                    self.knowledge_base['foods_to_avoid'][food_item.lower()] = {
                        'category': row.get('Food_Category', ''),
                        'risk': row.get('Health_Risk', ''),
                        'recommendation': row.get('Medical_Recommendation', ''),
                        'examples': row.get('Examples', '')
                    }
        
        # Process pregnancy diet clean (foods to eat)
        if not self.datasets['special_conditions']['pregnancy_diet_clean'].empty:
            df = self.datasets['special_conditions']['pregnancy_diet_clean']
            for _, row in df.iterrows():
                food_type = str(row.get('Type', '')).strip().upper()
                food_item = row.get('Food Item', '')
                if food_item:
                    if food_type == 'EAT':
                        self.knowledge_base['foods_to_eat'][food_item.lower()] = {
                            'food_group': row.get('Food Group', ''),
                            'nutrients': row.get('Nutrients', ''),
                            'benefit': row.get('Health Benefit', ''),
                            'trimester': row.get('Trimester', 'All'),
                            'remarks': row.get('Remarks', '')
                        }
                    elif food_type == 'AVOID':
                        self.knowledge_base['foods_to_avoid'][food_item.lower()] = {
                            'food_group': row.get('Food Group', ''),
                            'nutrients': row.get('Nutrients', ''),
                            'risk': row.get('Health Benefit', ''),
                            'trimester': row.get('Trimester', 'All'),
                            'remarks': row.get('Remarks', '')
                        }
        
        # Process do's and don'ts
        if not self.datasets['special_conditions']['dos_donts'].empty:
            df = self.datasets['special_conditions']['dos_donts']
            for _, row in df.iterrows():
                item_type = str(row.get('Type', '')).strip().upper()
                if 'DO' in item_type and "DON'T" not in item_type:
                    self.knowledge_base['dos_donts']['dos'].append({
                        'item': row.get('Item', ''),
                        'description': row.get('Description', ''),
                        'category': row.get('Category', ''),
                        'trimester': row.get('Trimester', ''),
                        'notes': row.get('Notes', '')
                    })
                elif "DON'T" in item_type or 'DONT' in item_type:
                    self.knowledge_base['dos_donts']['donts'].append({
                        'item': row.get('Item', ''),
                        'description': row.get('Description', ''),
                        'category': row.get('Category', ''),
                        'trimester': row.get('Trimester', ''),
                        'notes': row.get('Notes', '')
                    })
        
        # Process regional meal plans
        for region, df in self.datasets['regional_meals'].items():
            if not df.empty:
                self.knowledge_base['regional_foods'][region] = df.to_dict('records')
        
        # Process seasonal foods
        for season, df in self.datasets['seasonal'].items():
            if not df.empty:
                self.knowledge_base['seasonal_foods'][season] = df.to_dict('records')
        
        # Process trimester-specific plans
        for _, df in self.datasets['trimester_plans'].items():
            if not df.empty:
                for _, row in df.iterrows():
                    trimester_str = str(row.get('Trimester', '')).lower()
                    if '1' in trimester_str:
                        self.knowledge_base['trimester_specific'][1].append(row.to_dict())
                    elif '2' in trimester_str:
                        self.knowledge_base['trimester_specific'][2].append(row.to_dict())
                    elif '3' in trimester_str:
                        self.knowledge_base['trimester_specific'][3].append(row.to_dict())
    
    def is_pregnancy_related(self, question: str) -> bool:
        """
        Check if a question is related to pregnancy/maternal health.

        Returns True if question appears pregnancy-related, False for clearly
        off-topic queries (sports, weather, finance, etc.).
        """
        question_lower = question.lower()

        pregnancy_keywords = [
            "pregnancy", "pregnant", "baby", "trimester", "food", "eat", "diet",
            "nutrition", "maternal", "mother", "birth", "labor", "morning sickness",
            "prenatal", "postnatal", "breastfeed", "folic", "iron", "calcium",
            "vitamin", "health", "doctor", "hospital", "nausea", "contraction",
            "delivery", "lactation", "infant", "newborn", "fetus", "womb",
        ]

        off_topic_keywords = [
            "weather", "cricket", "football", "sports", "politics", "movie",
            "stock", "finance", "recipe", "coding", "programming", "computer",
            "phone", "travel", "hotel", "flight", "soccer", "basketball",
            "election", "actor", "singer", "game", "currency",
        ]

        if any(kw in question_lower for kw in pregnancy_keywords):
            return True

        if any(kw in question_lower for kw in off_topic_keywords):
            return False

        return True  # benefit of the doubt

    def needs_followup(self, question: str) -> Optional[dict]:
        """
        Check whether a question contains medical condition keywords that
        require clarification before answering.

        Returns a follow-up prompt dict or None.
        """
        question_lower = question.lower()

        if "diabetes" in question_lower:
            return {
                "question": (
                    "To give you the best advice, could you clarify: do you have "
                    "gestational diabetes (developed during pregnancy) or "
                    "pre-existing/Type 2 diabetes?"
                ),
                "context_key": "diabetes_type",
                "options": ["Gestational diabetes", "Pre-existing / Type 2 diabetes"],
            }

        if "blood pressure" in question_lower or "hypertension" in question_lower:
            return {
                "question": (
                    "Do you have pregnancy-induced hypertension or "
                    "pre-existing high blood pressure?"
                ),
                "context_key": "bp_type",
                "options": [
                    "Pregnancy-induced (preeclampsia)",
                    "Pre-existing hypertension",
                ],
            }

        if "thyroid" in question_lower:
            return {
                "question": (
                    "Are you dealing with hypothyroidism or hyperthyroidism "
                    "during pregnancy?"
                ),
                "context_key": "thyroid_type",
                "options": ["Hypothyroidism", "Hyperthyroidism"],
            }

        return None

    def classify_intent(self, question: str) -> str:
        """Classify user intent."""
        question_lower = question.lower()
        
        # Meal planning - higher priority
        if any(word in question_lower for word in ['meal plan', 'diet plan', 'what to eat', 'what should i eat', 'daily diet', 'menu', 'breakfast', 'lunch', 'dinner', 'snack']):
            return 'meal_plan'

        # Foods to eat (broader patterns - before safety_check)
        if any(phrase in question_lower for phrase in ['what foods', 'which foods', 'best foods']):
            return 'foods_to_eat'

        # Foods to avoid (broader patterns - before safety_check)
        if any(phrase in question_lower for phrase in ['what to avoid', 'which to avoid', 'not eat', "don't eat", 'dont eat']):
            return 'foods_to_avoid'

        # Safety questions
        if any(word in question_lower for word in ['can i eat', 'is it safe', 'should i avoid', 'can i have', 'safe to eat', 'okay to eat', 'is safe', 'good for', 'bad for']):
            return 'safety_check'

        # General info
        if any(phrase in question_lower for phrase in ['tell me about', 'information about', 'what is']):
            return 'general'

        # Portion / quantity questions
        if any(phrase in question_lower for phrase in ['how much', 'how many', 'portions', 'serving']):
            return 'portion_size'

        # Cooking tips
        if any(phrase in question_lower for phrase in ['cook', 'prepare', 'recipe']):
            return 'cooking_tips'
        
        # Foods to avoid (legacy patterns)
        if any(word in question_lower for word in ['avoid', "shouldn't eat", 'dangerous', 'foods to avoid', 'what not to']):
            return 'foods_to_avoid'
        
        # Benefits questions
        if any(word in question_lower for word in ['benefits', 'why eat', 'nutrients', 'nutritional', 'advantages', 'helps with', 'benefit of']):
            return 'benefits'
        
        # Trimester specific
        if any(word in question_lower for word in ['trimester', 'first trimester', 'second trimester', 'third trimester', '1st', '2nd', '3rd', '1 trimester', '2 trimester', '3 trimester']):
            return 'trimester_specific'
        
        # Seasonal
        if any(word in question_lower for word in ['summer', 'winter', 'monsoon', 'seasonal', 'season']):
            return 'seasonal'
        
        # Regional
        if any(word in question_lower for word in ['north indian', 'south indian', 'regional']):
            return 'regional'
        
        return 'general'
    
    def extract_keywords(self, question: str) -> List[str]:
        """Extract food and topic keywords from question."""
        keywords = []
        question_lower = question.lower()
        
        # Common food terms to check
        common_foods = [
            'egg', 'eggs', 'milk', 'fish', 'meat', 'chicken', 'vegetables', 'fruits', 
            'dairy', 'nuts', 'grains', 'rice', 'wheat', 'lentil', 'dal', 
            'papaya', 'pineapple', 'mango', 'banana', 'apple', 'spinach', 'orange',
            'yogurt', 'curd', 'cheese', 'paneer', 'ghee', 'butter',
            'potato', 'tomato', 'onion', 'garlic', 'ginger',
            'bread', 'roti', 'chapati', 'idli', 'dosa',
            'sweet', 'chocolate', 'coffee', 'tea', 'juice',
            'almond', 'walnut', 'cashew', 'dates', 'raisins',
            'salmon', 'tuna', 'shrimp', 'prawn',
            'mutton', 'lamb', 'pork', 'beef',
            # Additional common foods
            'avocado', 'quinoa', 'tofu', 'hummus', 'pomegranate', 'oats',
            'broccoli', 'carrot', 'cucumber', 'lettuce', 'kale',
            'strawberry', 'blueberry', 'watermelon', 'grapes',
            'brown rice', 'white rice', 'coconut', 'almond milk', 'soy milk',
            # Additional Indian foods
            'amla', 'neem', 'tulsi', 'moringa', 'drumstick', 'bitter gourd',
            'ash gourd', 'snake gourd', 'ridge gourd', 'bottle gourd',
            'fenugreek', 'methi', 'curry leaves', 'coriander', 'turmeric',
            'cumin', 'black sesame', 'white sesame', 'pumpkin seeds',
            'flax seeds', 'chia seeds', 'tamarind', 'kokum', 'jackfruit',
            'litchi', 'custard apple', 'wood apple', 'star fruit',
        ]
        
        # Check all foods in knowledge base
        for food in self.knowledge_base['foods_to_eat'].keys():
            if food in question_lower:
                keywords.append(food)
        
        for food in self.knowledge_base['foods_to_avoid'].keys():
            if food in question_lower:
                keywords.append(food)
        
        # Check common foods
        for food in common_foods:
            if food in question_lower and food not in keywords:
                keywords.append(food)
        
        # IMPROVED: Extract potential food names from question words
        # Skip common question words
        skip_words = {
            'can', 'could', 'should', 'would', 'is', 'are', 'was', 'were',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'shall',
            'eat', 'drink', 'consume', 'take', 'safe', 'okay', 'good', 'bad',
            'during', 'pregnancy', 'pregnant', 'trimester', 'women', 'woman',
            'about', 'what', 'which', 'when', 'where', 'who', 'how', 'why',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'with', 'from', 'of', 'by', 'as', 'this', 'that', 'these', 'those',
            'it', 'its', 'my', 'your', 'me', 'you', 'he', 'she', 'we', 'they'
        }
        
        # Extract words that might be food names (3+ letters, not in skip list)
        words = question_lower.replace('?', '').replace('!', '').replace(',', '').split()
        for word in words:
            if len(word) >= 3 and word not in skip_words and word not in keywords:
                # Likely a food name if not a common word
                keywords.append(word)
        
        # Remove duplicates and return
        return list(set(keywords))[:5]  # Limit to first 5 keywords
    
    def _paraphrase_query(self, question: str, keywords: List[str]) -> str:
        """
        Paraphrase user's question to show understanding.
        Makes the response feel more personal and shows the bot understood.
        
        Args:
            question: Original user question
            keywords: Extracted keywords from question
        
        Returns:
            Paraphrased understanding of the question
        """
        question_lower = question.lower()
        
        # Determine query intent
        if 'safe' in question_lower or 'can i' in question_lower or 'is it' in question_lower:
            action = 'safety'
        elif 'bad' in question_lower or 'avoid' in question_lower or 'don' in question_lower:
            action = 'avoid'
        elif 'benefit' in question_lower or 'good' in question_lower or 'help' in question_lower:
            action = 'benefits'
        elif 'meal' in question_lower or 'plan' in question_lower:
            action = 'meal_plan'
        else:
            action = 'general'
        
        # Build paraphrase with keywords
        if keywords:
            keyword_str = " and ".join([k.title() for k in keywords[:2]])
            
            if action == 'safety':
                paraphrase = f"You're asking about the safety of {keyword_str} during pregnancy"
            elif action == 'avoid':
                paraphrase = f"You want to know which foods to avoid, especially {keyword_str}"
            elif action == 'benefits':
                paraphrase = f"You're interested in the nutritional benefits of {keyword_str}"
            elif action == 'meal_plan':
                paraphrase = f"You're looking for meal planning ideas with {keyword_str}"
            else:
                paraphrase = f"You have questions about {keyword_str} in pregnancy nutrition"
        else:
            if action == 'safety':
                paraphrase = "You're asking about food safety during pregnancy"
            elif action == 'avoid':
                paraphrase = "You want to know which foods to avoid"
            elif action == 'benefits':
                paraphrase = "You're interested in nutritional benefits for pregnancy"
            elif action == 'meal_plan':
                paraphrase = "You're looking for a meal plan guide"
            else:
                paraphrase = "You have questions about pregnancy nutrition"
        
        return paraphrase
    
    def answer_question(self, question: str, trimester: Optional[int] = None, region: Optional[str] = None, season: Optional[str] = None) -> str:
        """
        Answer user question using all available datasets with improved response quality.
        
        Args:
            question: User's question
            trimester: Current trimester (1, 2, or 3)
            region: Regional preference (northveg, northnonveg, southveg, southnonveg)
            season: Current season (summer, winter, monsoon)
            
        Returns:
            Comprehensive, detailed answer from all datasets
        """
        intent = self.classify_intent(question)
        keywords = self.extract_keywords(question)
        
        answer_parts = []
        
        # Handle meal planning (doesn't require specific food keywords)
        if intent == 'meal_plan':
            answer_parts = self._handle_meal_plan_question(trimester, region, season, keywords)
        
        # Handle foods to avoid (doesn't require specific food keywords)
        elif intent == 'foods_to_avoid':
            answer_parts = self._handle_foods_to_avoid_question()
        
        # Handle trimester specific (doesn't require specific food keywords)
        elif intent == 'trimester_specific':
            answer_parts = self._handle_trimester_question(trimester)
        
        # Handle seasonal (doesn't require specific food keywords)
        elif intent == 'seasonal':
            answer_parts = self._handle_seasonal_question(season, keywords)
        
        # Handle portion size questions
        elif intent == 'portion_size':
            answer_parts = self._handle_portion_question(keywords)
        
        # Handle cooking tips questions
        elif intent == 'cooking_tips':
            answer_parts = self._handle_cooking_tips(keywords)
        
        # For other intents, check if we have keywords
        elif not keywords:
            return self._get_general_answer(intent, trimester)
        
        # Handle safety questions - most common and important
        elif intent == 'safety_check' or 'can i' in question.lower() or 'is it safe' in question.lower():
            answer_parts = self._handle_safety_question(keywords)
        
        # Handle benefits/nutrients
        elif intent == 'benefits':
            answer_parts = self._handle_benefits_question(keywords)
        
        # Default: comprehensive response
        else:
            answer_parts = self._handle_general_question_enhanced(keywords, trimester)
        
        if not answer_parts:
            # Multi-tier AI fallback chain (Gemini first for quality)
            if self._rate_limit_allows():
                context = self._create_default_context(trimester, region)
                
                # 1. Try Gemini first (fast, high quality)
                gemini_ans = self.gemini_ai.enhance_chatbot_response(question, context)
                if gemini_ans:
                    return f"🤖 **AI-Powered Answer:**\n\n{gemini_ans}\n\n💡 Note: This is AI-generated advice. Always consult your doctor for personalized guidance."
                
                # 2. Try LangChain + HuggingFace second
                langchain_ans = self.langchain_ai.generate_response(question, context)
                if langchain_ans:
                    return f"🤖 **AI-Powered Answer:**\n\n{langchain_ans}\n\n💡 Note: This is AI-generated advice. Always consult your doctor for personalized guidance."

                # 3. Try BERT+Flan-T5 last (local models, slower)
                if self.bert_flan_engine and self.bert_flan_engine.is_ready:
                    try:
                        all_knowledge = []
                        for category, items in self.knowledge_base.items():
                            if isinstance(items, dict):
                                for key, value in items.items():
                                    if isinstance(value, str):
                                        all_knowledge.append(f"{key}: {value}")
                                    elif isinstance(value, list):
                                        all_knowledge.extend([str(item) for item in value[:3]])
                            elif isinstance(items, list):
                                all_knowledge.extend([str(item) for item in items[:5]])
                        
                        relevant_texts = self.bert_flan_engine.semantic_search(
                            query=question,
                            knowledge_texts=all_knowledge[:100],
                            top_k=5
                        )
                        
                        context_str = "\n".join(relevant_texts)
                        bert_flan_answer = self.bert_flan_engine.generate_answer(
                            question=question,
                            context=context_str,
                            max_length=256
                        )
                        
                        if bert_flan_answer and len(bert_flan_answer.strip()) > 30:
                            return self._format_ai_response(bert_flan_answer, backend='bert_flan_t5')
                    except Exception as e:
                        print(f"⚠ BERT+Flan-T5 generation error: {e}")
            
            # Final fallback: Rule-based response from LangChain AI
            return self.langchain_ai.get_fallback_response(question)
        
        final_answer = '\n'.join(answer_parts)
        
        # Add pregnancy safety tip at the end
        if trimester:
            final_answer += f"\n\n💡 Tip: Always consult your doctor before making major dietary changes."
        
        # If final answer is poor quality or empty, try multi-tier AI fallback
        if self._is_poor_quality_answer(final_answer):
            if self._rate_limit_allows():
                context = self._create_default_context(trimester, region)
                
                # 1. Gemini first
                gemini_ans = self.gemini_ai.enhance_chatbot_response(question, context)
                if gemini_ans:
                    return self._format_ai_response(gemini_ans)
                
                # 2. LangChain second
                langchain_ans = self.langchain_ai.generate_response(question, context)
                if langchain_ans:
                    return self._format_ai_response(langchain_ans)

                # 3. BERT+Flan-T5 last
                if self.bert_flan_engine and self.bert_flan_engine.is_ready:
                    try:
                        all_knowledge = []
                        for category, items in self.knowledge_base.items():
                            if isinstance(items, dict):
                                for key, value in items.items():
                                    if isinstance(value, str):
                                        all_knowledge.append(f"{key}: {value}")
                                    elif isinstance(value, list):
                                        all_knowledge.extend([str(item) for item in value[:3]])
                            elif isinstance(items, list):
                                all_knowledge.extend([str(item) for item in items[:5]])
                        
                        relevant_texts = self.bert_flan_engine.semantic_search(
                            query=question,
                            knowledge_texts=all_knowledge[:100],
                            top_k=5
                        )
                        
                        context_str = "\n".join(relevant_texts)
                        bert_flan_answer = self.bert_flan_engine.generate_answer(
                            question=question,
                            context=context_str,
                            max_length=256
                        )
                        
                        if bert_flan_answer and len(bert_flan_answer.strip()) > 30:
                            return self._format_ai_response(bert_flan_answer, backend='bert_flan_t5')
                    except Exception as e:
                        print(f"⚠ BERT+Flan-T5 generation error: {e}")
            
            # Final fallback: Rule-based response from LangChain AI
            return self.langchain_ai.get_fallback_response(question)
        
        return final_answer

    def _handle_safety_question(self, keywords: List[str]) -> List[str]:
        """Handle safety check questions with detailed responses."""
        answer_parts = []
        safe_foods = []
        unsafe_foods = []
        
        for keyword in keywords:
            # Check if food should be avoided
            if keyword in self.knowledge_base['foods_to_avoid']:
                info = self.knowledge_base['foods_to_avoid'][keyword]
                unsafe_foods.append((keyword, info))
            # Check if food is safe
            elif keyword in self.knowledge_base['foods_to_eat']:
                info = self.knowledge_base['foods_to_eat'][keyword]
                safe_foods.append((keyword, info))
        
        # Check rate limit before proceeding with AI fallback
        if not self._rate_limit_allows():
            return ["⚠️ Rate limit reached. Please try again in a minute."]
        
        # If any unsafe foods found
        if unsafe_foods:
            answer_parts.append("⚠️ **FOODS TO AVOID:**\n")
            for food, info in unsafe_foods:
                answer_parts.append(f"❌ **{food.title()}**")
                if info.get('risk'):
                    answer_parts.append(f"   Reason: {info['risk']}")
                if info.get('recommendation'):
                    answer_parts.append(f"   Recommendation: {info['recommendation']}")
                if info.get('remarks'):
                    answer_parts.append(f"   Note: {info['remarks']}")
                if info.get('examples'):
                    answer_parts.append(f"   Examples: {info['examples']}")
                answer_parts.append("")  # Blank line
        
        # If safe foods found
        if safe_foods:
            answer_parts.append("\n✅ **SAFE & RECOMMENDED FOODS:**\n")
            for food, info in safe_foods:
                answer_parts.append(f"✓ **{food.title()}**")
                if info.get('nutrients'):
                    answer_parts.append(f"   Nutrients: {info['nutrients']}")
                if info.get('benefit'):
                    answer_parts.append(f"   Benefits: {info['benefit']}")
                if info.get('food_group'):
                    answer_parts.append(f"   Food Group: {info['food_group']}")
                if info.get('remarks'):
                    answer_parts.append(f"   💡 Tip: {info['remarks']}")
                answer_parts.append("")  # Blank line
        
        # If no specific info found in dataset, provide general guidance for common foods
        if not safe_foods and not unsafe_foods and keywords:
            answer_parts.append("**Food Safety Information**\n")
            for keyword in keywords:
                answer_parts.append(self._get_general_food_safety(keyword))
            return answer_parts
        
        # Add general safety tip
        if unsafe_foods or safe_foods:
            answer_parts.append("\n📌 **General Safety Tips:**")
            answer_parts.append("• Always wash fruits and vegetables thoroughly")
            answer_parts.append("• Cook meat, fish, and eggs completely")
            answer_parts.append("• Avoid unpasteurized dairy products")
            answer_parts.append("• Consult your doctor if you have specific concerns")
        
        return answer_parts
    
    def _get_general_food_safety(self, food: str) -> str:
        """Provide general safety guidance for common foods not in dataset."""
        food_lower = food.lower()
        
        # Fish guidance
        if food_lower in ['fish', 'salmon', 'tuna', 'shrimp', 'prawn', 'seafood']:
            return (
                f"**{food.title()}:**\n"
                "✅ **Generally Safe** (when cooked properly)\n"
                "   Benefits: High in protein, omega-3 fatty acids (DHA/EPA) for baby's brain development\n"
                "   ⚠️ **Important Tips:**\n"
                "   • Choose low-mercury fish: salmon, sardines, trout, anchovies\n"
                "   • Cook thoroughly - no raw/undercooked fish, sushi, or sashimi\n"
                "   • Limit to 2-3 servings per week (340g total)\n"
                "   • Avoid high-mercury fish: shark, swordfish, king mackerel, tilefish\n"
                "   • Fresh is best - avoid smoked or cured fish\n"
            )
        
        # Eggs guidance
        elif food_lower in ['egg', 'eggs']:
            return (
                f"**{food.title()}:**\n"
                "✅ **Safe when cooked**\n"
                "   Benefits: Excellent protein, choline for baby's brain development\n"
                "   ⚠️ **Important:**\n"
                "   • Always cook completely - both yolk and white should be firm\n"
                "   • Avoid raw/runny eggs, mayonnaise, raw cookie dough\n"
                "   • 1-2 eggs per day is safe and beneficial\n"
            )
        
        # Milk/dairy guidance
        elif food_lower in ['milk', 'dairy', 'cheese', 'yogurt', 'curd']:
            return (
                f"**{food.title()}:**\n"
                "✅ **Safe - Pasteurized only**\n"
                "   Benefits: Calcium, protein, vitamin D for bone development\n"
                "   ⚠️ **Important:**\n"
                "   • Only consume pasteurized dairy products\n"
                "   • Avoid soft cheeses (feta, brie, camembert) unless pasteurized\n"
                "   • 2-3 servings per day recommended\n"
            )
        
        # Meat guidance
        elif food_lower in ['meat', 'chicken', 'mutton', 'lamb', 'pork', 'beef']:
            return (
                f"**{food.title()}:**\n"
                "✅ **Safe when fully cooked**\n"
                "   Benefits: High-quality protein, iron, B vitamins\n"
                "   ⚠️ **Important:**\n"
                "   • Cook completely - no pink meat\n"
                "   • Use meat thermometer (165°F/74°C minimum)\n"
                "   • Avoid deli meats unless heated until steaming\n"
                "   • Good source of iron to prevent anemia\n"
            )
        
        # Fruits/vegetables
        elif food_lower in ['fruits', 'vegetables', 'fruit', 'vegetable']:
            return (
                f"**{food.title()}:**\n"
                "✅ **Highly recommended**\n"
                "   Benefits: Vitamins, minerals, fiber, antioxidants\n"
                "   ⚠️ **Important:**\n"
                "   • Wash thoroughly before eating\n"
                "   • Peel when possible\n"
                "   • 5+ servings per day recommended\n"
                "   • Variety is key\n"
            )
        
        # Try Gemini AI for unknown foods
        else:
            if self._rate_limit_allows():
                context = self._create_default_context()
                gemini_ans = self.gemini_ai.enhance_chatbot_response(f"Is {food} safe to eat during pregnancy? What are the benefits and risks?", context)
                if gemini_ans:
                    return f"**{food.title()}:**\n✅ AI Response:\n{gemini_ans}\n"

            return (
                f"**{food.title()}:**\n"
                "For specific safety information about this food, please consult your healthcare provider.\n"
                "\n**General Safety Guidelines:**\n"
                "• Cook all animal products thoroughly\n"
                "• Wash all produce\n"
                "• Avoid unpasteurized products\n"
                "• Choose fresh over processed when possible\n"
            )
    
    def _handle_foods_to_avoid_question(self) -> List[str]:
        """Handle foods to avoid questions with comprehensive list."""
        answer_parts = ["🚫 **COMPREHENSIVE LIST - FOODS TO AVOID DURING PREGNANCY**\n"]
        
        # Group by category
        by_category = {}
        for food, info in self.knowledge_base['foods_to_avoid'].items():
            category = info.get('category', info.get('food_group', 'General'))
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((food, info))
        
        # Display each category
        for category, foods in list(by_category.items())[:6]:
            answer_parts.append(f"\n**📌 {category}:**")
            for food, info in foods[:5]:
                risk = info.get('risk', info.get('Health_Risk', 'Not recommended'))
                answer_parts.append(f"  • **{food.title()}**: {risk}")
        
        # Add general guidelines
        answer_parts.append("\n**⚠️ General Guidelines:**")
        answer_parts.append("• Avoid all raw or undercooked meats, fish, and eggs")
        answer_parts.append("• Skip unpasteurized milk, cheese, and juices")
        answer_parts.append("• Limit high-mercury fish (shark, swordfish, king mackerel)")
        answer_parts.append("• No alcohol, smoking, or recreational drugs")
        answer_parts.append("• Wash all produce thoroughly before eating")
        
        return answer_parts
    
    def _handle_meal_plan_question(self, trimester: Optional[int], region: Optional[str], season: Optional[str], keywords: List[str]) -> List[str]:
        """Handle meal planning questions with detailed plans."""
        answer_parts = []
        
        if trimester and trimester in [1, 2, 3]:
            answer_parts.append(f"🍽️ **MEAL PLAN - TRIMESTER {trimester}**\n")
            
            # Get trimester-specific guidance
            trimester_guidance = {
                1: "Focus on managing morning sickness and building foundational nutrition with folic acid and B vitamins.",
                2: "This is the growth phase - increase calories (+300/day), protein, calcium, and iron intake.",
                3: "Prepare for delivery with high protein, fiber for digestion, and continued iron/calcium."
            }
            answer_parts.append(f"🎯 **Goal**: {trimester_guidance.get(trimester, '')}\n")
            
            # Get trimester-specific foods
            trimester_foods = self.knowledge_base['trimester_specific'].get(trimester, [])
            if trimester_foods:
                # Build sample meal plan
                answer_parts.append("**🌞 Sample Day Meal Plan:**\n")
                answer_parts.append("🍳 **Breakfast:**")
                answer_parts.append("  • Whole grain toast with peanut butter")
                answer_parts.append("  • Scrambled eggs or omelette")
                answer_parts.append("  • Fresh fruit (banana, apple, berries)")
                answer_parts.append("  • Glass of milk or fortified plant milk")
                answer_parts.append("")  
                
                answer_parts.append("🥗 **Mid-Morning Snack:**")
                answer_parts.append("  • Handful of nuts (almonds, walnuts)")
                answer_parts.append("  • Yogurt with honey")
                answer_parts.append("")  
                
                answer_parts.append("🍛 **Lunch:**")
                answer_parts.append("  • Brown rice or roti (2-3)")
                answer_parts.append("  • Dal (lentils) for protein")
                answer_parts.append("  • Seasonal vegetables curry")
                answer_parts.append("  • Salad with cucumber, tomato, carrots")
                answer_parts.append("  • Buttermilk or curd")
                answer_parts.append("")  
                
                answer_parts.append("🍏 **Evening Snack:**")
                answer_parts.append("  • Fruit smoothie or fresh juice")
                answer_parts.append("  • Whole grain crackers with cheese")
                answer_parts.append("")  
                
                answer_parts.append("🍲 **Dinner:**")
                answer_parts.append("  • Chapati or rice")
                answer_parts.append("  • Grilled fish/chicken or paneer")
                answer_parts.append("  • Leafy green vegetables (spinach, methi)")
                answer_parts.append("  • Light soup or dal")
                answer_parts.append("")  
        
        # Add regional or seasonal context if available
        if region:
            regional_meals = self.knowledge_base['regional_foods'].get(region, [])
            if regional_meals:
                answer_parts.append(f"\n**🌏 {region.title().replace('veg', ' Veg').replace('nonveg', ' Non-Veg')} Options:**")
                for meal in regional_meals[:5]:
                    meal_name = meal.get('meal') or meal.get('dish') or str(meal)
                    if meal_name:
                        answer_parts.append(f"  • {meal_name}")
        
        # Add general tips
        answer_parts.append("\n**💡 Meal Planning Tips:**")
        answer_parts.append("• Eat 5-6 small meals throughout the day")
        answer_parts.append("• Stay hydrated - drink 8-10 glasses of water")
        answer_parts.append("• Include variety from all food groups")
        answer_parts.append("• Take prenatal vitamins as prescribed")
        answer_parts.append("• Listen to your body and adjust portions")
        
        return answer_parts
    
    def _handle_trimester_question(self, trimester: Optional[int]) -> List[str]:
        """Handle trimester-specific questions."""
        answer_parts = []
        
        if not trimester or trimester not in [1, 2, 3]:
            return ["Please specify which trimester (1, 2, or 3) you'd like information for."]
        
        answer_parts.append(f"🤰 TRIMESTER {trimester} NUTRITION GUIDE\n")
        
        plans = self.knowledge_base['trimester_specific'].get(trimester, [])
        if plans:
            # Get top 8 suggestions
            for i, plan in enumerate(plans[:8], 1):
                answer_parts.append(f"  {i}. {plan}")
        
        return answer_parts
    
    def _handle_seasonal_question(self, season: Optional[str], keywords: List[str]) -> List[str]:
        """Handle seasonal diet questions with auto-detection of current season."""
        # Auto-detect season from current month if not provided
        if not season:
            month = _dt.datetime.now().month
            if 3 <= month <= 5:
                season = 'summer'
            elif 6 <= month <= 9:
                season = 'monsoon'
            else:
                season = 'winter'

        answer_parts = []

        seasonal_data = self.knowledge_base['seasonal_foods'].get(season, [])

        if seasonal_data:
            season_title = season.title()
            answer_parts.append(f"🌤️ **{season_title.upper()} PREGNANCY DIET**\n")
            answer_parts.append(f"Based on the current season ({season_title}), here are recommended foods:\n")

            for item in seasonal_data[:10]:
                if isinstance(item, dict):
                    food_name = (item.get('Food') or item.get('food') or
                                 item.get('Food Item') or item.get('Meal') or
                                 item.get('meal') or '')
                    benefit = (item.get('Benefit') or item.get('benefit') or
                               item.get('Benefits') or item.get('Health_Benefit') or '')
                    if food_name:
                        line = f"  • **{food_name}**"
                        if benefit:
                            line += f": {benefit}"
                        answer_parts.append(line)
                    else:
                        # Fall back to raw representation
                        answer_parts.append(f"  • {item}")

        if not answer_parts:
            seasonal_meals = self.unified_loader.get_meals_by_preference(season=season)
            if seasonal_meals:
                answer_parts.append(f"🌤️ **{season.title()} MEAL IDEAS**:\n")
                for meal in seasonal_meals[:8]:
                    meal_name = (meal.get('meal') or meal.get('dish') or
                                 meal.get('food') or meal.get('name'))
                    if meal_name:
                        answer_parts.append(f"  • {meal_name}")

        if not answer_parts:
            # Generic seasonal advice
            seasonal_tips = {
                'summer': [
                    "• Hydrating fruits: watermelon, muskmelon, cucumber",
                    "• Cooling foods: coconut water, buttermilk, lassi",
                    "• Fresh salads with cucumber, tomato, and mint",
                    "• Light meals – avoid heavy oily food",
                    "• Drink 10–12 glasses of water daily",
                ],
                'monsoon': [
                    "• Cooked warm foods to prevent infections",
                    "• Ginger tea to soothe nausea",
                    "• Soups and stews with vegetables",
                    "• Avoid raw/street food due to contamination risk",
                    "• Include turmeric and ginger for immunity",
                ],
                'winter': [
                    "• Warm soups: tomato, spinach, lentil",
                    "• Seasonal vegetables: carrots, beets, sweet potato",
                    "• Dry fruits: almonds, walnuts, dates for warmth",
                    "• Warm milk with turmeric at night",
                    "• Ghee in moderation for warmth and nutrition",
                ],
            }
            tips = seasonal_tips.get(season, [])
            if tips:
                answer_parts.append(f"🌤️ **{season.title()} Pregnancy Diet Tips:**\n")
                answer_parts.extend(tips)

        return answer_parts

    def _handle_portion_question(self, keywords: List[str]) -> List[str]:
        """Handle portion and serving-size questions."""
        answer_parts = ["📏 **PORTION & SERVING SIZE GUIDE FOR PREGNANCY**\n"]

        portion_info = {
            'milk': "3 cups (720 ml) per day – rich in calcium and vitamin D",
            'dairy': "3 servings per day (1 cup milk / 1 cup yogurt / 45g cheese)",
            'eggs': "1–2 eggs per day – excellent protein and choline source",
            'fish': "2–3 servings (340g total) per week – choose low-mercury varieties",
            'meat': "2–3 servings (85–100g per serving) per day",
            'chicken': "85–100g per serving, 2–3 servings per day",
            'fruits': "2–4 servings per day (1 medium fruit = 1 serving)",
            'vegetables': "5+ servings per day (1 cup raw / ½ cup cooked = 1 serving)",
            'grains': "6–8 servings per day (1 slice bread / ½ cup cooked rice = 1 serving)",
            'rice': "½ cup cooked = 1 serving; 2–3 servings per day",
            'nuts': "30g (handful) per day – almonds, walnuts, cashews",
            'almonds': "8–10 almonds per day",
            'dates': "3–5 dates per day, especially in third trimester",
        }

        found_any = False
        for keyword in keywords:
            if keyword in portion_info:
                found_any = True
                answer_parts.append(f"• **{keyword.title()}**: {portion_info[keyword]}")
            elif keyword in self.knowledge_base['foods_to_eat']:
                found_any = True
                info = self.knowledge_base['foods_to_eat'][keyword]
                answer_parts.append(f"• **{keyword.title()}**: {info.get('remarks', 'Consume in moderate portions.')}")

        if not found_any:
            answer_parts.append("**General Portion Guidelines:**")
            answer_parts.append("• Dairy: 3 cups/day")
            answer_parts.append("• Protein (meat/eggs/legumes): 2–3 servings/day")
            answer_parts.append("• Fruits: 2–4 servings/day")
            answer_parts.append("• Vegetables: 5+ servings/day")
            answer_parts.append("• Grains: 6–8 servings/day")
            answer_parts.append("• Fats/Oils: Use sparingly – 2–3 teaspoons/day")

        answer_parts.append("\n💡 Tip: Eat smaller, more frequent meals to manage nausea and heartburn.")

        return answer_parts

    def _handle_cooking_tips(self, keywords: List[str]) -> List[str]:
        """Handle cooking method / preparation questions."""
        answer_parts = ["👩‍🍳 **SAFE COOKING TIPS DURING PREGNANCY**\n"]

        generic_tips = [
            "• **Always cook meat and poultry thoroughly** – internal temp ≥74°C (165°F)",
            "• **Cook fish until opaque** – internal temp ≥63°C (145°F)",
            "• **Cook eggs until both yolk and white are firm**",
            "• **Wash all fruits and vegetables** under running water",
            "• **Avoid raw sprouts** – rinse and cook them",
            "• **Use separate cutting boards** for raw meat and produce",
            "• **Refrigerate leftovers** within 2 hours of cooking",
            "• **Avoid reheating food more than once**",
        ]

        food_specific = {
            'fish': [
                "• Grill, bake, or steam – avoid deep frying",
                "• Marinate with lemon and herbs for flavour",
                "• Never serve raw or undercooked",
            ],
            'chicken': [
                "• Bake, boil, or grill – avoid heavy oil",
                "• Internal temp must reach 74°C",
                "• Remove skin to reduce saturated fat",
            ],
            'egg': [
                "• Hard-boil or scramble – avoid sunny-side up",
                "• Avoid raw egg in batters or dressings",
            ],
            'eggs': [
                "• Hard-boil or scramble – avoid sunny-side up",
                "• Avoid raw egg in batters or dressings",
            ],
            'vegetables': [
                "• Steam to preserve nutrients",
                "• Avoid overcooking leafy greens",
                "• Wash thoroughly before cooking",
            ],
        }

        found_specific = False
        for keyword in keywords:
            if keyword in food_specific:
                found_specific = True
                answer_parts.append(f"\n**{keyword.title()} – Cooking Tips:**")
                answer_parts.extend(food_specific[keyword])

        answer_parts.append("\n**🔥 General Safe Cooking Practices:**")
        answer_parts.extend(generic_tips)

        return answer_parts

    def _handle_general_question_enhanced(self, keywords: List[str], trimester: Optional[int]) -> List[str]:
        """
        Enhanced general question handler:
        1. Searches foods_to_eat and foods_to_avoid knowledge base.
        2. Searches all loaded DataFrames for keyword matches.
        3. Falls through to AI if nothing found.
        """
        answer_parts = []
        found_any = False

        for keyword in keywords:
            if keyword in self.knowledge_base['foods_to_eat']:
                found_any = True
                info = self.knowledge_base['foods_to_eat'][keyword]
                answer_parts.append(f"✓ **{keyword.title()}** *(Safe to eat)*")
                if info.get('nutrients'):
                    answer_parts.append(f"  🔬 Nutrients: {info['nutrients']}")
                if info.get('benefit'):
                    answer_parts.append(f"  💪 Benefits: {info['benefit']}")
                if info.get('remarks'):
                    answer_parts.append(f"  💡 Note: {info['remarks']}")
                answer_parts.append("")

            elif keyword in self.knowledge_base['foods_to_avoid']:
                found_any = True
                info = self.knowledge_base['foods_to_avoid'][keyword]
                answer_parts.append(f"⚠️ **{keyword.title()}** *(Caution – often avoided)*")
                if info.get('risk'):
                    answer_parts.append(f"  ❗ Risk: {info['risk']}")
                if info.get('recommendation'):
                    answer_parts.append(f"  📌 Advice: {info['recommendation']}")
                answer_parts.append("")

        # Search DataFrames for additional matches
        if not found_any:
            for category, dataset_group in self.datasets.items():
                for name, df in dataset_group.items():
                    if df.empty:
                        continue
                    for keyword in keywords:
                        try:
                            mask = df.apply(
                                lambda col: col.astype(str).str.contains(
                                    keyword, case=False, na=False
                                )
                            ).any(axis=1)
                            matches = df[mask].head(3)
                            for _, row in matches.iterrows():
                                food_col = (row.get('Food Item') or row.get('Food') or
                                            row.get('food') or row.get('meal') or
                                            row.get('Meal') or keyword)
                                answer_parts.append(f"• **{str(food_col).title()}** (from {name})")
                                found_any = True
                        except Exception:
                            pass
                if found_any:
                    break

        # Fall through to AI if still nothing
        if not found_any and keywords and self._rate_limit_allows():
            for keyword in keywords[:3]:
                context = self._create_default_context(trimester)
                gemini_info = self.gemini_ai.enhance_chatbot_response(
                    f"Tell me about {keyword} during pregnancy - is it safe? What are the benefits?",
                    context,
                )
                if gemini_info:
                    answer_parts.append(f"🤖 **{keyword.title()}** (AI-Powered):")
                    answer_parts.append(gemini_info)
                    answer_parts.append("")
                    found_any = True

        return answer_parts
    
    def _handle_benefits_question(self, keywords: List[str]) -> List[str]:
        """Handle questions about food benefits with detailed information."""
        answer_parts = ["🌟 **NUTRITIONAL BENEFITS & RECOMMENDATIONS**\n"]
        
        found_any = False
        for keyword in keywords:
            if keyword in self.knowledge_base['foods_to_eat']:
                info = self.knowledge_base['foods_to_eat'][keyword]
                found_any = True
                answer_parts.append(f"\n**{keyword.title()}:**")
                
                if info.get('nutrients'):
                    answer_parts.append(f"  🔬 Nutrients: {info['nutrients']}")
                if info.get('benefit'):
                    answer_parts.append(f"  💪 Benefits: {info['benefit']}")
                if info.get('food_group'):
                    answer_parts.append(f"  📁 Food Group: {info['food_group']}")
                if info.get('trimester'):
                    answer_parts.append(f"  🤰 Best for: Trimester {info['trimester']}")
                if info.get('remarks'):
                    answer_parts.append(f"  💡 Note: {info['remarks']}")
                answer_parts.append("")  # Blank line
            elif self._rate_limit_allows():
                context = self._create_default_context()
                gemini_ans = self.gemini_ai.enhance_chatbot_response(f"What are the nutritional benefits and risks of eating {keyword} during pregnancy?", context)
                if gemini_ans:
                    found_any = True
                    answer_parts.append(f"\n**{keyword.title()} (AI-Powered):**")
                    answer_parts.append(f"{gemini_ans}")
                    answer_parts.append("")
        
        if not found_any:
            # Provide general nutrition benefits
            answer_parts.append("**Key Nutrients for Pregnancy:**\n")
            answer_parts.append("• **Folic Acid**: Prevents neural tube defects (leafy greens, fortified cereals)")
            answer_parts.append("• **Iron**: Prevents anemia, supports blood production (lean meat, beans, spinach)")
            answer_parts.append("• **Calcium**: Builds baby's bones and teeth (dairy, fortified foods)")
            answer_parts.append("• **Protein**: Supports baby's growth (eggs, meat, legumes, nuts)")
            answer_parts.append("• **DHA/Omega-3**: Brain development (fish, walnuts, flaxseed)")
            answer_parts.append("• **Vitamin D**: Bone health, immunity (sunlight, fortified milk)")
            answer_parts.append("• **Vitamin C**: Iron absorption, immunity (citrus, berries)")
        
        return answer_parts
    
    def _handle_general_question(self, keywords: List[str], trimester: Optional[int]) -> List[str]:
        """Handle general questions about pregnancy nutrition."""
        answer_parts = []
        found_any = False
        
        for keyword in keywords:
            if keyword in self.knowledge_base['foods_to_eat']:
                found_any = True
                info = self.knowledge_base['foods_to_eat'][keyword]
                answer_parts.append(f"✓ {keyword.title()}")
                answer_parts.append(f"  Nutrients: {info.get('nutrients', 'Various minerals and vitamins')}")
                answer_parts.append(f"  Benefit: {info.get('benefit', 'Supports pregnancy health')}\n")
        
        # If no food found in database, try Gemini AI for each keyword
        if not found_any and keywords and self._rate_limit_allows():
            for keyword in keywords[:3]:  # Limit to first 3 keywords
                context = self._create_default_context()
                gemini_info = self.gemini_ai.enhance_chatbot_response(f"Tell me about {keyword} during pregnancy - is it safe? What are the benefits?", context)
                if gemini_info:
                    answer_parts.append(f"🤖 **{keyword.title()}** (AI-Powered):")
                    answer_parts.append(gemini_info)
                    answer_parts.append("")
        
        return answer_parts
    
    def _get_general_answer(self, intent: str, trimester: Optional[int]) -> str:
        """Provide helpful general answer when specifics aren't available."""
        general_responses = {
            'safety_check': "For safety questions about specific foods, please mention the food name and I'll provide detailed information about whether it's safe during pregnancy.",
            'foods_to_avoid': "During pregnancy, generally avoid: raw/undercooked meats, unpasteurized dairy, high-mercury fish, raw eggs, and unwashed produce. Ask me about specific foods!",
            'meal_plan': f"I can help create a meal plan for you! Tell me your preferences (vegetarian/non-vegetarian), region, and current trimester (you're in Trimester {trimester or '?'})",
            'trimester_specific': "Each trimester has specific nutritional needs. Tell me your current trimester and I can provide tailored recommendations.",
            'seasonal': "Seasonal diets help maximize fresh produce benefits. Which season are you in (summer/winter/monsoon)?",
            'benefits': "Ask about specific foods and I'll tell you their nutritional benefits and why they're important during pregnancy.",
            'general': "I'm your pregnancy nutrition expert! Ask me about:\n  • Food safety (can I eat...?)\n  • Meal plans\n  • Nutritional benefits\n  • Trimester-specific needs\n  • Seasonal diets"
        }
        
        return general_responses.get(intent, general_responses['general'])

    def get_dos_donts_answer(self, topic: str, trimester: Optional[int] = None) -> Dict:
        """
        Get structured Do's and Don'Ts answer for a specific topic.
        
        Args:
            topic: Food item or topic to query
            trimester: Current trimester (1, 2, or 3)
            
        Returns:
            Dictionary with 'dos' and 'donts' lists
        """
        topic_lower = topic.lower().strip()
        
        dos = []
        donts = []
        
        # Search in database
        found = False
        
        # Check foods to eat (DO's)
        if topic_lower in self.knowledge_base['foods_to_eat']:
            found = True
            info = self.knowledge_base['foods_to_eat'][topic_lower]
            dos.append({
                'item': topic.title(),
                'reason': f"Safe & beneficial. {info.get('benefit', 'Provides important nutrients.')}",
                'details': {
                    'nutrients': info.get('nutrients', 'Various minerals and vitamins'),
                    'food_group': info.get('food_group', 'General'),
                    'trimester': info.get('trimester', 'All trimesters'),
                    'tip': info.get('remarks', 'Ensure well-cooked and properly prepared.')
                }
            })
        
        # Check foods to avoid (DON'T's)
        if topic_lower in self.knowledge_base['foods_to_avoid']:
            found = True
            info = self.knowledge_base['foods_to_avoid'][topic_lower]
            donts.append({
                'item': topic.title(),
                'reason': info.get('risk', 'May cause health risks during pregnancy'),
                'details': {
                    'recommendation': info.get('recommendation', 'Best to avoid'),
                    'risk_level': info.get('category', 'Medium'),
                    'safer_alternative': f"Choose cooked or pasteurized alternatives",
                    'tip': info.get('remarks', 'Consult your doctor if consumed.')
                }
            })
        
        # If not found in structured data, search in dos_donts dataset
        if not found and self.knowledge_base['dos_donts']['dos']:
            # Search dos
            for do_item in self.knowledge_base['dos_donts']['dos']:
                if topic_lower in str(do_item.get('item', '')).lower():
                    dos.append({
                        'item': do_item.get('item', topic),
                        'reason': do_item.get('description', 'Good for pregnancy'),
                        'details': {
                            'category': do_item.get('category', 'General'),
                            'trimester': do_item.get('trimester', 'All'),
                            'notes': do_item.get('notes', '')
                        }
                    })
                    found = True
            
            # Search donts
            for dont_item in self.knowledge_base['dos_donts']['donts']:
                if topic_lower in str(dont_item.get('item', '')).lower():
                    donts.append({
                        'item': dont_item.get('item', topic),
                        'reason': dont_item.get('description', 'Should be avoided'),
                        'details': {
                            'category': dont_item.get('category', 'General'),
                            'trimester': dont_item.get('trimester', 'All'),
                            'notes': dont_item.get('notes', '')
                        }
                    })
                    found = True
        
        # If still not found, provide general DO's and DON'Ts
        if not found:
            dos, donts = self._get_general_dos_donts(topic)
        
        return {
            'dos': dos,
            'donts': donts,
            'found': found,
            'source': 'database' if found else 'general_guidelines'
        }

    def _get_general_dos_donts(self, topic: str) -> tuple:
        """Get general Do's and Don'Ts for common pregnancy foods."""
        topic_lower = topic.lower()
        
        general_guidelines = {
            'fish': {
                'dos': [
                    {'item': 'Low-mercury fish', 'reason': 'Safe and beneficial for DHA/Omega-3', 'details': {'examples': 'Salmon, sardines, trout, anchovies', 'frequency': '2-3 servings per week'}},
                    {'item': 'Well-cooked fish', 'reason': 'Eliminates harmful bacteria', 'details': {'cooking': 'Bake, boil, or grill until opaque', 'temp': '63°C minimum'}},
                    {'item': 'Fresh fish', 'reason': 'Better nutritional value', 'details': {'storage': 'Store in refrigerator, use within 2 days'}}
                ],
                'donts': [
                    {'item': 'High-mercury fish', 'reason': 'Mercury can harm fetal development', 'details': {'avoid': 'Shark, swordfish, king mackerel, tilefish'}},
                    {'item': 'Raw/undercooked fish', 'reason': 'Risk of bacteria and parasites', 'details': {'avoid': 'Sushi, sashimi, ceviche, smoked fish'}},
                    {'item': 'Excessive portions', 'reason': 'Cumulative mercury exposure', 'details': {'limit': 'Max 340g (12 oz) per week'}}
                ]
            },
            'papaya': {
                'dos': [
                    {'item': 'Ripe papaya', 'reason': 'Safe and rich in vitamin C', 'details': {'color': 'Yellow, soft to touch', 'benefits': 'Improves digestion and immunity'}},
                    {'item': 'Cooked/steamed ripe papaya', 'reason': 'Extra safety measure', 'details': {'cooking': 'Light cooking is safe for ripe papaya'}}
                ],
                'donts': [
                    {'item': 'Unripe papaya', 'reason': 'Contains latex which can trigger contractions', 'details': {'risk': 'May cause miscarriage', 'identification': 'Green, hard texture'}},
                    {'item': 'Excessive amounts', 'reason': 'Too much vitamin A can be harmful', 'details': {'limit': 'Moderate portions only'}},
                    {'item': 'Papaya seeds/leaves', 'reason': 'Unknown safety profile in pregnancy', 'details': {'avoid': 'Only eat the flesh of ripe fruit'}}
                ]
            },
            'egg': {
                'dos': [
                    {'item': 'Fully cooked eggs', 'reason': 'Excellent source of choline for baby brain development', 'details': {'cooking': 'Both yolk and white must be firm', 'benefit': '1-2 eggs per day is safe'}},
                    {'item': 'Boiled/scrambled eggs', 'reason': 'Safest cooking methods', 'details': {'methods': 'Boil, scramble, bake - avoid runny yolks'}}
                ],
                'donts': [
                    {'item': 'Raw/undercooked eggs', 'reason': 'Risk of Salmonella infection', 'details': {'avoid': 'Raw cookie dough, mayonnaise, soft-boiled eggs'}},
                    {'item': 'Raw egg-based foods', 'reason': 'Contamination risk', 'details': {'avoid': 'Homemade ice cream, hollandaise, caesar dressing'}}
                ]
            },
            'milk': {
                'dos': [
                    {'item': 'Pasteurized milk', 'reason': 'Rich in calcium and vitamin D for bone development', 'details': {'amount': '3 cups per day recommended', 'benefits': 'Supports baby skeleton development'}},
                    {'item': 'Milk products (pasteurized)', 'reason': 'Excellent calcium sources', 'details': {'examples': 'Pasteurized cheese, yogurt, paneer, fortified milk'}}
                ],
                'donts': [
                    {'item': 'Unpasteurized/raw milk', 'reason': 'Risk of Listeria and other harmful bacteria', 'details': {'risk': 'Can cause miscarriage or stillbirth'}},
                    {'item': 'Soft cheeses (unless pasteurized)', 'reason': 'May contain Listeria', 'details': {'avoid': 'Feta, brie, camembert, queso fresco'}},
                    {'item': 'Aged raw-milk cheeses', 'reason': 'Higher bacterial risk', 'details': {'avoid': 'Check label for pasteurization'}}
                ]
            }
        }
        
        # Check if topic matches any guidelines
        for food, guidelines in general_guidelines.items():
            if food in topic_lower:
                dos_list = [{'item': d['item'], 'reason': d['reason'], 'details': d['details']} for d in guidelines['dos']]
                donts_list = [{'item': d['item'], 'reason': d['reason'], 'details': d['details']} for d in guidelines['donts']]
                return dos_list, donts_list
        
        # Default response if topic not found
        dos_default = [
            {'item': 'Consult with your doctor', 'reason': 'Get personalized medical advice', 'details': {'about': topic}},
            {'item': 'Ensure proper cooking', 'reason': 'Most foods are safe when properly prepared', 'details': {'temperatures': 'Cook meat to 73°C, fish to 63°C, eggs until firm'}}
        ]
        
        donts_default = [
            {'item': 'Avoid unprepared foods', 'reason': 'Unknown foods should be verified by doctor', 'details': {'risk': 'Potential contamination or allergen risk'}}
        ]
        
        return dos_default, donts_default

    def answer_question_structured(
        self,
        question: str,
        trimester: Optional[int] = None,
        context_answers: Optional[Dict] = None,
        force_ai: bool = False,
    ) -> Dict:
        """
        FAST & SMART: Answer question with optional Do's and Don'Ts format.
        INCLUDES: Query reflection (paraphrased understanding) + Direct answer
        Uses: 1. Fast cache lookup (instant), 2. AI model fallback (fast), 3. Conditional Do's/Don'ts
        
        Do's/Don'ts are ONLY generated for:
        - Safety questions (Can I eat X?)
        - Avoidance questions (What to avoid?)
        - Guidance questions (What's safe/unsafe?)
        
        NOT generated for:
        - Benefits questions (just list benefits)
        - Meal planning (provide meal ideas)
        - Nutritional info (just numbers)
        - Recipe requests (just recipes)
        
        Args:
            question: User's question
            trimester: Current trimester
            context_answers: Optional dict of clarified context (e.g. {"diabetes_type": "Gestational"})
            force_ai: If True, skip the response cache and call AI directly
            
        Returns:
            Dictionary with query_reflection, answer text, and optional dos/donts lists
        """
        import time
        start_time = time.time()
        
        # Check response cache first (instant) – skip when force_ai is set
        cache_key = f"{question.lower().strip()}_{trimester or 'any'}"
        if not force_ai and cache_key in self._response_cache:
            cached_response = self._response_cache[cache_key]
            # Check if cache is still valid (TTL not expired)
            cache_time = cached_response.get('_cache_time', 0)
            if time.time() - cache_time < self._cache_ttl:
                cached_response['response_time'] = time.time() - start_time
                cached_response['from_cache'] = True
                return cached_response
        
        
        # Extract keywords and classify intent
        keywords = self.extract_keywords(question)
        intent = self.classify_intent(question)
        
        # CREATE QUERY REFLECTION - Show bot understood the question
        query_reflection = self._paraphrase_query(question, keywords)
        
        # If context_answers are provided, incorporate them into the Gemini prompt
        if context_answers and self.gemini_ai.available and self._rate_limit_allows():
            context_str = "; ".join(f"{k}: {v}" for k, v in context_answers.items())
            enriched_question = f"{question} [Context: {context_str}]"
            context = self._create_default_context(trimester)
            gemini_ans = self.gemini_ai.enhance_chatbot_response(enriched_question, context)
            if gemini_ans:
                response_time = time.time() - start_time
                return {
                    'query_reflection': query_reflection,
                    'answer': self._format_ai_response(gemini_ans, backend='gemini'),
                    'dos': [],
                    'donts': [],
                    'keywords': keywords,
                    'intent': intent,
                    'source': 'ai_model',
                    'response_time': response_time,
                    'from_cache': False,
                }

        # If force_ai, go directly to AI
        if force_ai and self.gemini_ai.available and self._rate_limit_allows():
            context = self._create_default_context(trimester)
            gemini_ans = self.gemini_ai.enhance_chatbot_response(question, context)
            if gemini_ans:
                response_time = time.time() - start_time
                return {
                    'query_reflection': query_reflection,
                    'answer': self._format_ai_response(gemini_ans, backend='gemini'),
                    'dos': [],
                    'donts': [],
                    'keywords': keywords,
                    'intent': intent,
                    'source': 'ai_model',
                    'response_time': response_time,
                    'from_cache': False,
                }

        # DETERMINE IF DO'S/DON'TS FORMAT IS NEEDED
        needs_dos_donts = intent in ['safety_check', 'foods_to_avoid', 'trimester_specific', 'general']
        
        # STEP 1: FAST - Try instant cache lookup for each keyword
        dos_final = []
        donts_final = []
        cached_answer = None
        source = 'ai_model'
        
        # ONLY generate Do's/Don'ts if the query needs it
        if needs_dos_donts:
            for keyword in keywords[:3]:
                # Try SUPER FAST cache lookup (milliseconds)
                cache_result = self.unified_loader.quick_answer_from_cache(keyword)
                
                if cache_result['found']:
                    source = 'database_cache'
                    data = cache_result['data']
                    
                    # Convert cache data to Do's & Don'Ts format
                    if cache_result['type'] == 'food':
                        dos_final.append({
                            'item': keyword.title(),
                            'reason': 'Found in database - safe food',
                            'details': {key: value for key, value in data.items() if not key.startswith('source_')}
                        })
                    elif cache_result['type'] == 'dos_donts':
                        dos_final.append({
                            'item': data.get('item', keyword).title(),
                            'reason': data.get('description', 'Food guidance'),
                            'details': data
                        })
        
        # If we found cached data with Do's/Don'ts, build answer from it
        if needs_dos_donts and source == 'database_cache' and (dos_final or donts_final):
            cached_answer = self.answer_question(question, trimester)
            
            # Check if cached_answer is actually useful (not generic fallback)
            # If answer is poor quality, skip the cache and use AI instead
            if self._is_poor_quality_answer(cached_answer):
                source = 'ai_model'  # Reset source to try AI instead
            else:
                response_time = time.time() - start_time
                return {
                    'query_reflection': query_reflection,
                    'answer': cached_answer,
                    'dos': dos_final,
                    'donts': donts_final,
                    'keywords': keywords,
                    'intent': intent,
                    'source': source,
                    'response_time': response_time,
                    'from_cache': True
                }
        
        # STEP 2: FALLBACK - If not in cache, use AI model for FAST answer
        response_time_ai_start = time.time()
        
        # Get AI answer in structured format
        answer_text = self.answer_question(question, trimester)
        
        # ONLY generate Do's and Don'Ts if the query type needs it
        if needs_dos_donts:
            # If no keywords found, extract from question
            if not keywords:
                keywords = [keyword.strip() for keyword in question.split() if len(keyword) > 3]
            
            # Generate Do's and Don'Ts from AI using the extracted keywords
            for keyword in keywords[:2]:
                dos_donts = self.get_dos_donts_answer(keyword, trimester)
                dos_final.extend(dos_donts['dos'][:1])
                donts_final.extend(dos_donts['donts'][:1])
        
        response_time = time.time() - start_time
        
        result = {
            'query_reflection': query_reflection,
            'answer': answer_text,
            'dos': dos_final,
            'donts': donts_final,
            'keywords': keywords,
            'intent': intent,
            'source': 'ai_model',
            'response_time': response_time,
            'from_cache': False,
            '_cache_time': time.time()
        }
        
        # Store in cache for future requests (not when force_ai)
        if not force_ai:
            self._response_cache[cache_key] = result.copy()
        
        return result

    def answer_question_regenerate(
        self, question: str, trimester: Optional[int] = None
    ) -> Dict:
        """
        Force a fresh Gemini AI answer (bypass dataset cache) for regeneration
        after negative user feedback.

        Returns:
            Dictionary with answer, query_reflection, dos, donts, source keys.
        """
        return self.answer_question_structured(
            question=question,
            trimester=trimester,
            force_ai=True,
        )

    def quick_answer(self, question: str, trimester: Optional[int] = None) -> str:
        """
        LIGHTNING FAST: Get quick answer from cache or AI (< 1 second).
        
        Args:
            question: User's question
            trimester: Current trimester
            
        Returns:
            Quick text answer
        """
        keywords = self.extract_keywords(question)
        
        # Try FAST cache lookup first
        for keyword in keywords[:1]:
            cache_result = self.unified_loader.quick_answer_from_cache(keyword)
            if cache_result['found']:
                return f"Quick Answer: {keyword.title()} found in database."
        
        # Fallback to AI
        return self.answer_question(question, trimester)



_comprehensive_chatbot_instance = None


def get_comprehensive_chatbot() -> ComprehensiveChatbot:
    """Get or initialize the comprehensive chatbot (singleton)."""
    global _comprehensive_chatbot_instance
    if _comprehensive_chatbot_instance is None:
        _comprehensive_chatbot_instance = ComprehensiveChatbot()
    return _comprehensive_chatbot_instance


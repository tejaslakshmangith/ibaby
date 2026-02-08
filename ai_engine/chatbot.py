"""Enhanced AI Chatbot Module with improved Q&A."""
import re
from typing import Dict, List, Tuple, Optional


class MaternalFoodChatbot:
    """
    Enhanced AI Chatbot for maternal food recommendations.
    Provides comprehensive, well-structured answers for pregnancy nutrition.
    """
    
    def __init__(self):
        """Initialize the chatbot."""
        self._bert_tokenizer = None
        self._bert_model = None
        self._flan_tokenizer = None
        self._flan_model = None
        self._models_loaded = False
        self._knowledge_base = self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self) -> Dict:
        """Initialize comprehensive knowledge base for pregnancy nutrition."""
        return {
            'trimester_needs': {
                1: {
                    'focus': 'Folic acid, B vitamins, nausea management',
                    'key_nutrients': ['Folic Acid', 'Vitamin B6', 'Ginger', 'Vitamin B12'],
                    'critical': 'Neural tube development, morning sickness relief'
                },
                2: {
                    'focus': 'Calcium, Iron, Protein, energy needs',
                    'key_nutrients': ['Calcium', 'Iron', 'Protein', 'Vitamin D'],
                    'critical': 'Baby growth, bone development, anemia prevention'
                },
                3: {
                    'focus': 'Protein, Fiber, Antioxidants, preparation for labor',
                    'key_nutrients': ['Protein', 'Fiber', 'Iron', 'Dates'],
                    'critical': 'Labor preparation, digestive health, energy'
                }
            },
            'common_conditions': {
                'morning_sickness': ['Ginger', 'Lemon', 'Small frequent meals', 'Crackers'],
                'constipation': ['Fiber-rich foods', 'Prunes', 'Water', 'Leafy greens'],
                'anemia': ['Iron-rich foods', 'Spinach', 'Meat', 'Fortified cereals'],
                'gestational_diabetes': ['Complex carbs', 'Lean protein', 'Low sugar', 'Whole grains'],
                'high_blood_pressure': ['Low sodium', 'Potassium-rich', 'Calcium', 'No spicy foods']
            }
        }
    
    def _load_models(self):
        """Load BERT + Flan-T5 models via background engine."""
        if self._models_loaded:
            return
        try:
            from ai_engine.bert_flan_engine import get_engine
            engine = get_engine()
            self._bert_tokenizer = engine._bert_tokenizer
            self._bert_model = engine._bert_model
            self._flan_tokenizer = engine._flan_tokenizer
            self._flan_model = engine._flan_model
            self._models_loaded = engine.is_ready
        except Exception:
            self._models_loaded = True  # Use rule-based fallback
    
    def classify_intent(self, question: str) -> str:
        """
        Classify user intent with more granular categories.
        
        Args:
            question: User's question
            
        Returns:
            Intent category
        """
        question_lower = question.lower()
        
        # Safety and restrictions (high priority)
        if any(word in question_lower for word in ['can i eat', 'is it safe', 'safe to eat', 'should i avoid', 'avoid', 'dangerous', 'risky']):
            return 'safety_check'
        
        # Precautions and warnings
        elif any(word in question_lower for word in ['precaution', 'warning', 'risk', 'danger', 'when to avoid']):
            return 'precautions'
        
        # Benefits and advantages
        elif any(word in question_lower for word in ['benefits', 'good for', 'why eat', 'advantages', 'helpful', 'useful']):
            return 'benefits'
        
        # Nutritional content
        elif any(word in question_lower for word in ['nutrition', 'nutrients', 'vitamins', 'minerals', 'protein', 'calcium', 'iron', 'nutritional', 'contains']):
            return 'nutritional_info'
        
        # Quantity and serving
        elif any(word in question_lower for word in ['how much', 'quantity', 'how many', 'serving', 'portion', 'amount', 'grams']):
            return 'quantity'
        
        # Preparation and recipes
        elif any(word in question_lower for word in ['how to', 'prepare', 'cook', 'recipe', 'cooking', 'preparation', 'make', 'best way']):
            return 'preparation'
        
        # Trimester-specific
        elif any(word in question_lower for word in ['trimester', 'first trimester', 'second trimester', 'third trimester', '1st trimester', '2nd trimester', '3rd trimester', 'during pregnancy']):
            return 'trimester_specific'
        
        # Health conditions
        elif any(word in question_lower for word in ['morning sickness', 'nausea', 'constipation', 'anemia', 'diabetes', 'blood pressure', 'acidity', 'heartburn', 'swelling']):
            return 'health_condition'
        
        # Seasonal
        elif any(word in question_lower for word in ['summer', 'winter', 'monsoon', 'seasonal', 'season']):
            return 'seasonal'
        
        # Cravings and preferences
        elif any(word in question_lower for word in ['craving', 'want', 'like', 'prefer', 'can i have', 'what about']):
            return 'cravings'
        
        # General questions
        else:
            return 'general'
    
    def extract_food_entities(self, question: str, all_foods: List) -> List:
        """
        Extract food items mentioned in the question.
        
        Args:
            question: User's question
            all_foods: List of FoodItem objects from database
            
        Returns:
            List of matching FoodItem objects
        """
        question_lower = question.lower()
        matched_foods = []
        
        for food in all_foods:
            # Exact match with English name
            if food.name_english.lower() in question_lower:
                if food not in matched_foods:
                    matched_foods.append(food)
                continue
            
            # Exact match with Hindi name
            if food.name_hindi and food.name_hindi.lower() in question_lower:
                if food not in matched_foods:
                    matched_foods.append(food)
                continue
            
            # Partial word matches for compound words
            words = question_lower.split()
            for word in words:
                if len(word) > 3:  # Skip very short words
                    if word in food.name_english.lower() or (food.name_hindi and word in food.name_hindi.lower()):
                        if food not in matched_foods:
                            matched_foods.append(food)
                        break
        
        return matched_foods
    
    def generate_response(self, question: str, intent: str, foods: List, trimester: int = 1) -> str:
        """
        Generate comprehensive response to the user's question.
        
        Args:
            question: User's question
            intent: Classified intent
            foods: List of relevant FoodItem objects
            trimester: User's current trimester
            
        Returns:
            Generated response
        """
        # If no foods found, give comprehensive general response
        if not foods:
            return self._generate_comprehensive_general_response(question, intent, trimester)
        
        # For single food, give detailed comprehensive response
        if len(foods) == 1:
            return self._generate_comprehensive_single_food_response(foods[0], intent, trimester, question)
        
        # For multiple foods, give comparative response
        return self._generate_comparative_food_response(foods, intent, trimester)
    
    def _generate_comprehensive_general_response(self, question: str, intent: str, trimester: int) -> str:
        """Generate comprehensive response when no specific food is mentioned."""
        
        if intent == 'safety_check':
            return f"""🛡️ **Food Safety in Trimester {trimester}**

During your {self._get_trimester_name(trimester)}, food safety is crucial. Here's what you should know:

**Foods to Focus On:**
✅ Fresh fruits and vegetables (wash thoroughly)
✅ Cooked meat and fish (avoid raw)
✅ Pasteurized dairy products
✅ Whole grains and legumes
✅ Nuts and seeds (unsalted preferred)

**Foods to Avoid:**
❌ Raw or undercooked meat
❌ Raw seafood and sushi
❌ Unpasteurized dairy
❌ High-mercury fish
❌ Raw eggs and soft cheeses

**Which specific food are you concerned about?** I can provide detailed safety information for any food item."""
        
        elif intent == 'benefits':
            focus = self._knowledge_base['trimester_needs'][trimester]['focus']
            return f"""💪 **Nutritional Benefits in Trimester {trimester}**

During this phase, your body needs:
**Key Focus Areas:** {focus}

**Food Groups & Their Benefits:**
🥬 **Leafy Greens** - Iron, calcium, folic acid
🥛 **Dairy** - Calcium for bone development
🍗 **Protein** - Muscle and tissue development
🍎 **Fruits** - Vitamins and antioxidants
🌾 **Whole Grains** - Energy and fiber
🥚 **Eggs** - Choline for brain development
🥜 **Legumes** - Plant-based protein and iron

**Which food would you like to know the specific benefits of?**"""
        
        elif intent == 'nutritional_info':
            key_nutrients = self._knowledge_base['trimester_needs'][trimester]['key_nutrients']
            return f"""🔬 **Important Nutrients for Trimester {trimester}**

**Focus on these nutrients:**
• {', '.join(key_nutrients)}

**Why these matter:**
{self._knowledge_base['trimester_needs'][trimester]['critical']}

**Daily Recommended Amounts:**
- Calcium: 1000 mg
- Iron: 27 mg
- Folic Acid: 600-800 mcg
- Protein: 71 grams
- Calories: +300-500 kcal

**Which specific nutrient or food would you like to know about?**"""
        
        elif intent == 'health_condition':
            return f"""🏥 **Managing Common Pregnancy Conditions**

I can help with:
- Morning sickness and nausea
- Constipation and digestion
- Anemia and low iron
- Gestational diabetes
- High blood pressure
- Heartburn and acidity

**What specific condition are you experiencing?** I can recommend foods that help manage it."""
        
        elif intent == 'seasonal':
            return f"""🌤️ **Seasonal Food Recommendations for Pregnancy**

Different seasons offer unique nutritional benefits. Here are some seasonal considerations:

**Summer (Hot Weather):**
- Stay hydrated with watermelon, cucumber, and coconut water
- Focus on cooling foods like yogurt and mint
- Include seasonal fruits: mangoes, melons, berries

**Winter (Cold Weather):**
- Warm soups and stews with root vegetables
- Citrus fruits for vitamin C: oranges, lemons
- Nuts and seeds for healthy fats

**Monsoon (Rainy Season):**
- Ginger and turmeric for immunity
- Avoid raw foods that may cause infections
- Focus on cooked vegetables and fermented foods

**Which season are you asking about?** I can provide specific recommendations for your current season."""
        
        else:  # General
            return f"""👋 **Welcome to Your Pregnancy Nutrition Guide!**

I'm here to help answer your questions about pregnancy nutrition. You can ask me about:

✅ **Safety** - Is it safe to eat during pregnancy?
✅ **Benefits** - What are the health benefits?
✅ **Nutrition** - What nutrients does it contain?
✅ **Preparation** - How should I cook/prepare it?
✅ **Quantity** - How much should I eat?
✅ **Precautions** - Any warnings or precautions?
✅ **Health Issues** - Help managing pregnancy conditions

**Try asking:** "Can I eat papaya during pregnancy?" or "What are the benefits of milk?"""
    
    def _get_trimester_name(self, trimester: int) -> str:
        """Get trimester name."""
        names = {1: 'First Trimester', 2: 'Second Trimester', 3: 'Third Trimester'}
        return names.get(trimester, f'Trimester {trimester}')
    
    def _generate_comprehensive_single_food_response(self, food, intent: str, trimester: int, question: str) -> str:
        """Generate detailed comprehensive response for a single food item."""
        
        trimester_suit = food.get_trimester_suitability()
        trimester_key = f'trimester_{trimester}'
        is_safe = trimester_suit.get(trimester_key, True)
        
        response = f"🍽️ **{food.name_english}** ({food.name_hindi or 'N/A'})\n\n"
        
        # Add category and origin
        response += f"📍 Category: {food.category} | Region: {food.regional_origin or 'All India'}\n\n"
        
        if intent == 'safety_check':
            response += self._generate_safety_section(food, is_safe, trimester)
            response += self._generate_precautions_section(food)
            response += self._generate_benefits_summary(food)
        
        elif intent == 'benefits':
            response += self._generate_benefits_section(food, trimester)
            response += self._generate_safety_indicator(is_safe, trimester)
        
        elif intent == 'nutritional_info':
            response += self._generate_nutrition_section(food)
            response += self._generate_benefits_summary(food)
        
        elif intent == 'preparation':
            response += self._generate_preparation_section(food)
            response += self._generate_quantity_recommendations(food)
        
        elif intent == 'quantity':
            response += self._generate_quantity_section(food)
            response += self._generate_preparation_section(food)
        
        elif intent == 'precautions':
            response += self._generate_precautions_section(food)
            response += self._generate_safety_indicator(is_safe, trimester)
        
        elif intent == 'trimester_specific':
            response += self._generate_trimester_guidance(food, trimester, is_safe)
        
        elif intent == 'health_condition':
            response += self._generate_condition_help(food, question, is_safe, trimester)
        
        else:  # General overview
            response += self._generate_safety_indicator(is_safe, trimester)
            response += "\n\n"
            response += self._generate_benefits_summary(food)
            response += "\n"
            response += self._generate_nutrition_summary(food)
            response += "\n"
            if food.preparation_tips:
                response += f"\n**How to Prepare:**\n{food.preparation_tips}\n"
        
        return response.strip()
    
    def _generate_safety_section(self, food, is_safe: bool, trimester: int) -> str:
        """Generate safety section."""
        if is_safe:
            return f"""✅ **SAFE DURING TRIMESTER {trimester}**

{food.name_english} is generally considered safe to consume during your {self._get_trimester_name(trimester)}.

"""
        else:
            return f"""⚠️ **USE WITH CAUTION IN TRIMESTER {trimester}**

It's recommended to avoid or limit {food.name_english} during your {self._get_trimester_name(trimester)}.

"""
    
    def _generate_precautions_section(self, food) -> str:
        """Generate precautions section."""
        if food.precautions:
            return f"""🚨 **Precautions & Important Notes:**
{food.precautions}

"""
        return ""
    
    def _generate_benefits_section(self, food, trimester: int) -> str:
        """Generate benefits section."""
        if food.benefits:
            return f"""💪 **Health Benefits:**
{food.benefits}

"""
        return f"""💪 **General Benefits:**
{food.name_english} provides essential nutrients important for maternal health and fetal development.

"""
    
    def _generate_benefits_summary(self, food) -> str:
        """Generate brief benefits summary."""
        if food.benefits:
            return f"""**Benefits:** {food.benefits}"""
        return ""
    
    def _generate_nutrition_section(self, food) -> str:
        """Generate detailed nutrition section."""
        nutrition = food.get_nutritional_info()
        
        response = """🔬 **Nutritional Composition (per 100g):**\n
"""
        if nutrition:
            # Organize nutrients
            categories = {
                'Energy': ['calories', 'energy'],
                'Proteins': ['protein'],
                'Carbohydrates': ['carbohydrate', 'carbs'],
                'Fats': ['fat', 'fats'],
                'Minerals': ['calcium', 'iron', 'phosphorus', 'magnesium', 'zinc'],
                'Vitamins': ['vitamin a', 'vitamin b', 'vitamin c', 'vitamin d'],
                'Fiber': ['fiber', 'dietary fiber']
            }
            
            for category, keywords in categories.items():
                found = False
                for nutrient, value in nutrition.items():
                    if any(kw in nutrient.lower() for kw in keywords):
                        if not found:
                            response += f"\n**{category}:**\n"
                            found = True
                        response += f"• {nutrient.replace('_', ' ').title()}: {value}\n"
        else:
            response += "Nutritional data from reliable sources. Check packaging or consult a nutritionist for exact values.\n"
        
        return response + "\n"
    
    def _generate_nutrition_summary(self, food) -> str:
        """Generate brief nutrition summary."""
        nutrition = food.get_nutritional_info()
        if nutrition:
            items = list(nutrition.items())[:3]
            summary = ", ".join([f"{k}: {v}" for k, v in items])
            return f"**Nutrition (per 100g):** {summary}"
        return ""
    
    def _generate_preparation_section(self, food) -> str:
        """Generate preparation section."""
        if food.preparation_tips:
            return f"""👨‍🍳 **How to Prepare:**
{food.preparation_tips}

"""
        return ""
    
    def _generate_quantity_recommendations(self, food) -> str:
        """Generate quantity recommendations based on food type."""
        quantity_guides = {
            'Vegetables': '1-2 cups (100-200g) per meal',
            'Fruits': '1-2 servings (100-150g) per day',
            'Dairy': '2-3 servings (200-300ml) per day',
            'Meat': '75-100g per serving',
            'Legumes': '75-100g (cooked) per serving',
            'Grains': '45-60g per serving',
            'Nuts': '25-30g per day'
        }
        
        category = food.category or 'Food'
        recommended = quantity_guides.get(category, f'Follow portion control guidelines for {category}')
        
        return f"""📏 **Recommended Quantity:**
{recommended}

"""
    
    def _generate_quantity_section(self, food) -> str:
        """Generate quantity section."""
        return self._generate_quantity_recommendations(food)
    
    def _generate_safety_indicator(self, is_safe: bool, trimester: int) -> str:
        """Generate safety indicator."""
        if is_safe:
            return f"✅ **Status:** Safe for Trimester {trimester}\n\n"
        return f"⚠️ **Status:** Use caution in Trimester {trimester}\n\n"
    
    def _generate_trimester_guidance(self, food, trimester: int, is_safe: bool) -> str:
        """Generate trimester-specific guidance."""
        response = f"""📅 **Trimester {trimester} Guidance:**

"""
        if is_safe:
            response += f"✅ {food.name_english} is beneficial during your {self._get_trimester_name(trimester)}.\n\n"
        else:
            response += f"⚠️ Avoid or limit {food.name_english} during your {self._get_trimester_name(trimester)}.\n\n"
        
        response += f"{food.benefits or 'Rich in essential nutrients for fetal development.'}\n\n"
        
        if food.precautions:
            response += f"**Important:** {food.precautions}\n"
        
        return response
    
    def _generate_condition_help(self, food, question: str, is_safe: bool, trimester: int) -> str:
        """Generate help for health conditions."""
        response = f"""🏥 **For Your Health Condition:**

"""
        
        if 'morning' in question.lower() or 'nausea' in question.lower():
            response += f"For morning sickness: Eat small portions of {food.name_english}, as tolerated.\n"
        
        elif 'constipation' in question.lower():
            response += f"For constipation: {food.name_english} may help due to its dietary content.\n"
        
        elif 'anemia' in question.lower():
            response += f"For anemia: {food.name_english} is rich in iron and can help increase your iron levels.\n"
        
        response += f"\n**Safety:** {'✅ Safe' if is_safe else '⚠️ Use caution'} during {self._get_trimester_name(trimester)}\n"
        
        if food.preparation_tips:
            response += f"\n**Best Preparation:** {food.preparation_tips}\n"
        
        return response
    
    def _generate_comparative_food_response(self, foods: List, intent: str, trimester: int) -> str:
        """Generate response comparing multiple foods."""
        response = f"🔍 **Found {len(foods)} Related Foods:**\n\n"
        
        for i, food in enumerate(foods[:5], 1):  # Limit to 5 foods
            trimester_suit = food.get_trimester_suitability()
            trimester_key = f'trimester_{trimester}'
            is_safe = trimester_suit.get(trimester_key, True)
            
            response += f"{i}. **{food.name_english}** ({food.name_hindi or 'N/A'})\n"
            
            if is_safe:
                response += "   ✅ Safe for your trimester\n"
            else:
                response += "   ⚠️ Use with caution\n"
            
            if intent == 'benefits':
                if food.benefits:
                    brief = food.benefits[:80] + ("..." if len(food.benefits) > 80 else "")
                    response += f"   {brief}\n"
            
            elif intent == 'nutritional_info':
                nutrition = food.get_nutritional_info()
                if nutrition:
                    key_nutrients = list(nutrition.items())[:1]
                    for nutrient, value in key_nutrients:
                        response += f"   • {nutrient}: {value}\n"
            
            response += "\n"
        
        if len(foods) > 5:
            response += f"...and {len(foods) - 5} more foods. \n\n"
        
        response += f"**Would you like detailed information about any specific food?**"
        
        return response.strip()
    
    def answer_question(self, question: str, all_foods: List, trimester: int = 1) -> Dict:
        """
        Main method to answer a user question with comprehensive response.
        
        Args:
            question: User's question
            all_foods: List of all FoodItem objects from database
            trimester: User's current trimester
            
        Returns:
            Dictionary with answer and metadata
        """
        # Load models if needed (lazy loading)
        if not self._models_loaded:
            try:
                self._load_models()
            except Exception as e:
                print(f"Continuing without AI models: {e}")
        
        # Classify intent
        intent = self.classify_intent(question)
        
        # Extract food entities
        foods = self.extract_food_entities(question, all_foods)
        
        # Generate comprehensive response
        response = self.generate_response(question, intent, foods, trimester)
        
        return {
            'answer': response,
            'intent': intent,
            'foods_mentioned': [food.name_english for food in foods],
            'trimester': trimester,
            'confidence': 'high' if foods else 'medium'
        }
    
    def get_suggested_questions(self, trimester: int = 1) -> List[str]:
        """
        Get suggested questions based on trimester.
        
        Args:
            trimester: Current trimester
            
        Returns:
            List of suggested questions
        """
        suggestions = {
            1: [
                "Can I eat papaya during first trimester?",
                "What foods help with morning sickness?",
                "Is milk safe during pregnancy?",
                "What's the benefit of spinach for pregnancy?",
                "Can I eat eggs during early pregnancy?"
            ],
            2: [
                "What foods are best for second trimester growth?",
                "Can I eat dates during pregnancy?",
                "Benefits of almonds during pregnancy",
                "How should I prepare lentils for maximum nutrition?",
                "Is yogurt good for my baby's development?"
            ],
            3: [
                "Foods that help prepare for labor naturally",
                "Can I eat papaya in third trimester?",
                "How much ghee should I consume daily?",
                "Why are dates recommended for pregnancy?",
                "What foods help prevent constipation?"
            ]
        }
        return suggestions.get(trimester, suggestions[1])


# Global chatbot instance (singleton)
_chatbot_instance = None


def get_chatbot():
    """Get or create chatbot instance (singleton pattern)."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = MaternalFoodChatbot()
    return _chatbot_instance

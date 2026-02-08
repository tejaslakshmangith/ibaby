"""AI-Powered Meal Plan Generator using Unified Dataset Integration and Gemini AI."""
import random
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ai_engine.unified_dataset_loader import UnifiedDatasetLoader
from ai_engine.gemini_integration import GeminiNutritionAI


class MealPlanner:
    """Generate personalized meal plans for pregnant women using comprehensive datasets and AI."""
    
    def __init__(self, db, unified_loader: Optional[UnifiedDatasetLoader] = None):
        """
        Initialize the meal planner with unified dataset integration and AI support.
        
        Args:
            db: Database instance
            unified_loader: UnifiedDatasetLoader instance (creates new if not provided)
        """
        self.db = db
        self.unified_loader = unified_loader or UnifiedDatasetLoader()
        self.gemini_ai = GeminiNutritionAI()
        self.meal_types = ['breakfast', 'mid_morning_snack', 'lunch', 'evening_snack', 'dinner']
    
    def generate_meal_plan(
        self,
        user,
        days: int = 7,
        region: Optional[str] = None,
        diet_type: Optional[str] = None,
        trimester: Optional[int] = None,
        special_conditions: Optional[List[str]] = None,
        meal_frequency: str = '3meals'
    ) -> Dict:
        """
        Generate a personalized meal plan using user preferences and all datasets.
        
        Args:
            user: User object with preferences
            days: Number of days for the meal plan (1-30)
            region: Regional preference (required)
            diet_type: Diet type - 'veg', 'nonveg', 'vegan' (required)
            trimester: Pregnancy trimester 1-3 (required)
            special_conditions: List of conditions like 'diabetes', 'gestational_diabetes'
            meal_frequency: '3meals' (breakfast, lunch, dinner) or '5meals' (add snacks)
        
        Returns:
            {
                'meal_plan': [...],  # Daily meal plans
                'nutrition_summary': {...},  # Overall nutrition stats
                'table_format': [...],  # Formatted for display
                'data_sources_used': [...],  # Which datasets were used
                'error': '...' (if error occurred)
            }
        """
        try:
            # Validate required preferences
            if not all([region, diet_type, trimester]):
                return {
                    'error': 'Required preferences: region, diet_type, trimester'
                }
            
            # Normalize inputs
            region = self._normalize_region(region)
            diet_type = self._normalize_diet(diet_type)
            
            # Determine which meal types to generate
            if meal_frequency == '5meals':
                meal_types = self.meal_types  # All 5 meals
            else:
                meal_types = ['breakfast', 'lunch', 'dinner']  # 3 main meals
            
            # Track used meals to ensure variety (avoid immediate repetition)
            used_meals_tracker = {meal_type: set() for meal_type in meal_types}
            
            # Generate daily meal plans
            meal_plan = []
            data_sources_used = set()
            nutrition_totals = {
                'calories': 0,
                'protein': 0,
                'carbs': 0,
                'fat': 0,
                'fiber': 0,
                'iron': 0,
                'calcium': 0,
                'folic_acid': 0,
                'vitamin_a': 0,
                'vitamin_b6': 0,
                'vitamin_b12': 0,
                'vitamin_c': 0,
                'vitamin_d': 0,
                'zinc': 0,
                'magnesium': 0,
                'omega3': 0
            }
            
            for day in range(1, days + 1):
                day_meals = self._generate_day_meals(
                    user=user,
                    day=day,
                    meal_types=meal_types,
                    region=region,
                    diet_type=diet_type,
                    trimester=trimester,
                    special_conditions=special_conditions or [],
                    data_sources_used=data_sources_used,
                    used_meals_tracker=used_meals_tracker
                )
                
                if 'error' in day_meals:
                    return day_meals
                
                meal_plan.append(day_meals)
                
                # Accumulate nutrition
                day_nutrition = self._calculate_day_nutrition(day_meals['meals'])
                for key in nutrition_totals:
                    nutrition_totals[key] += day_nutrition.get(key, 0)
            
            # Calculate averages and targets
            pregnancy_targets = self._get_pregnancy_nutrition_targets(trimester)
            nutrition_summary = self._generate_nutrition_summary(
                nutrition_totals,
                pregnancy_targets,
                days,
                meal_frequency
            )
            
            # Check if AI enhancement is available and add recommendations
            ai_enhanced = False
            if self.gemini_ai.available:
                # Use AI to validate nutrition and provide recommendations
                ai_nutrition = self.gemini_ai.calculate_advanced_nutrition(
                    [day['meals'] for day in meal_plan if 'meals' in day]
                )
                if ai_nutrition:
                    ai_enhanced = True
                    # Merge AI nutrition data if more comprehensive
                    for nutrient, value in ai_nutrition.items():
                        if nutrient not in nutrition_totals or nutrition_totals[nutrient] == 0:
                            nutrition_totals[nutrient] = value
            
            # Format for table display
            table_format = self._format_meal_plan_table(meal_plan)
            
            return {
                'meal_plan': meal_plan,
                'nutrition_summary': nutrition_summary,
                'table_format': table_format,
                'data_sources_used': list(data_sources_used),
                'ai_enhanced': ai_enhanced,
                'preferences': {
                    'region': region,
                    'diet_type': diet_type,
                    'trimester': trimester,
                    'special_conditions': special_conditions,
                    'meal_frequency': meal_frequency
                }
            }
        
        except Exception as e:
            print(f"Error generating meal plan: {e}")
            return {'error': str(e)}
    
    def _generate_day_meals(
        self,
        user,
        day: int,
        meal_types: List[str],
        region: str,
        diet_type: str,
        trimester: int,
        special_conditions: List[str],
        data_sources_used: set,
        used_meals_tracker: Dict[str, set]
    ) -> Dict:
        """Generate meals for a single day with variety tracking."""
        try:
            day_meals = {
                'day': day,
                'date': (datetime.now() + timedelta(days=day-1)).strftime('%Y-%m-%d'),
                'meals': {}
            }
            
            for meal_type in meal_types:
                # Get meals from unified loader with multiple filter attempts
                meals = self._get_meal_for_type(
                    meal_type=meal_type,
                    region=region,
                    diet_type=diet_type,
                    trimester=trimester,
                    special_conditions=special_conditions
                )
                
                if meals:
                    # Filter out recently used meals for variety (for multi-day plans)
                    available_meals = [m for m in meals if self._get_meal_id(m) not in used_meals_tracker[meal_type]]
                    
                    # If all meals have been used recently, reset tracker and use all meals
                    if not available_meals:
                        used_meals_tracker[meal_type].clear()
                        available_meals = meals
                    
                    selected_meal = random.choice(available_meals)
                    day_meals['meals'][meal_type] = selected_meal
                    
                    # Track this meal to avoid immediate repetition
                    used_meals_tracker[meal_type].add(self._get_meal_id(selected_meal))
                    
                    # Keep tracker size manageable (last 3 meals for each type)
                    if len(used_meals_tracker[meal_type]) > 3:
                        # Remove oldest entry (convert to list, remove first, convert back)
                        tracker_list = list(used_meals_tracker[meal_type])
                        used_meals_tracker[meal_type] = set(tracker_list[1:])
                    
                    # Track data source
                    category = selected_meal.get('source_category', 'unknown')
                    if category:
                        data_sources_used.add(category)
                else:
                    # Fallback to any meal matching diet and region
                    fallback_meals = self.unified_loader.get_meals_by_preference(
                        region=region,
                        diet_type=diet_type
                    )
                    if fallback_meals:
                        # Apply same variety logic to fallback meals
                        available_fallback = [m for m in fallback_meals if self._get_meal_id(m) not in used_meals_tracker[meal_type]]
                        if not available_fallback:
                            used_meals_tracker[meal_type].clear()
                            available_fallback = fallback_meals
                        
                        selected_meal = random.choice(available_fallback)
                        day_meals['meals'][meal_type] = selected_meal
                        used_meals_tracker[meal_type].add(self._get_meal_id(selected_meal))
                        data_sources_used.add('regional_fallback')
                    else:
                        # Return error - no meals available
                        return {
                            'error': f'No meals available for {meal_type} with your preferences'
                        }
            
            return day_meals
        
        except Exception as e:
            print(f"Error generating day {day} meals: {e}")
            return {'error': f'Error generating meals for day {day}: {str(e)}'}
    
    def _get_meal_id(self, meal: Dict) -> str:
        """Get unique identifier for a meal to track usage."""
        # Try multiple columns to identify the meal
        for col in ['food', 'food_item', 'meal', 'dish', 'dish_name', 'recipe', 'name', 'item']:
            if col in meal and meal[col]:
                return str(meal[col]).strip().lower()
        # Fallback: use hash of meal dict (less ideal but works)
        return str(hash(frozenset(meal.items())))
    
    def _get_meal_for_type(
        self,
        meal_type: str,
        region: str,
        diet_type: str,
        trimester: int,
        special_conditions: List[str]
    ) -> List[Dict]:
        """Get suitable meals for specific meal type with all preferences."""
        
        # Try to get meals matching all preferences (including condition if applicable)
        condition = special_conditions[0] if special_conditions else None
        
        # Try without season (season parameter removed)
        meals = self.unified_loader.get_meals_by_preference(
            region=region,
            diet_type=diet_type,
            trimester=trimester,
            condition=condition,
            meal_type=meal_type
        )
        
        # If no exact match, try without condition
        if not meals and condition:
            meals = self.unified_loader.get_meals_by_preference(
                region=region,
                diet_type=diet_type,
                trimester=trimester,
                meal_type=meal_type
            )
        
        # If still no match, try without trimester
        if not meals:
            meals = self.unified_loader.get_meals_by_preference(
                region=region,
                diet_type=diet_type,
                condition=condition
            )
        
        # Last resort - any meal matching region and diet
        if not meals:
            meals = self.unified_loader.get_meals_by_preference(
                region=region,
                diet_type=diet_type
            )
        
        return meals
    
    def _calculate_day_nutrition(self, meals: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate nutrition values for all meals in a day with comprehensive nutrients."""
        nutrition = {
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'fiber': 0,
            'iron': 0,
            'calcium': 0,
            'folic_acid': 0,
            'vitamin_a': 0,
            'vitamin_b6': 0,
            'vitamin_b12': 0,
            'vitamin_c': 0,
            'vitamin_d': 0,
            'zinc': 0,
            'magnesium': 0,
            'omega3': 0
        }
        
        for meal_type, meal in meals.items():
            # Try to extract nutrition from meal
            for nutrient in nutrition:
                value = meal.get(nutrient, 0)
                if isinstance(value, (int, float)):
                    nutrition[nutrient] += value
        
        return nutrition
    
    def _get_pregnancy_nutrition_targets(self, trimester: int) -> Dict[str, Dict]:
        """Get recommended nutrition targets by trimester with comprehensive nutrients."""
        targets = {
            1: {
                'calories': {'daily': 2000, 'unit': 'kcal'},
                'protein': {'daily': 50, 'unit': 'g'},
                'carbs': {'daily': 175, 'unit': 'g'},
                'fat': {'daily': 70, 'unit': 'g'},
                'fiber': {'daily': 25, 'unit': 'g'},
                'calcium': {'daily': 1000, 'unit': 'mg'},
                'iron': {'daily': 27, 'unit': 'mg'},
                'folic_acid': {'daily': 600, 'unit': 'mcg'},
                'vitamin_a': {'daily': 770, 'unit': 'mcg'},
                'vitamin_b6': {'daily': 1.9, 'unit': 'mg'},
                'vitamin_b12': {'daily': 2.6, 'unit': 'mcg'},
                'vitamin_c': {'daily': 85, 'unit': 'mg'},
                'vitamin_d': {'daily': 15, 'unit': 'mcg'},
                'zinc': {'daily': 11, 'unit': 'mg'},
                'magnesium': {'daily': 350, 'unit': 'mg'},
                'omega3': {'daily': 300, 'unit': 'mg'}
            },
            2: {
                'calories': {'daily': 2300, 'unit': 'kcal'},
                'protein': {'daily': 60, 'unit': 'g'},
                'carbs': {'daily': 175, 'unit': 'g'},
                'fat': {'daily': 70, 'unit': 'g'},
                'fiber': {'daily': 25, 'unit': 'g'},
                'calcium': {'daily': 1000, 'unit': 'mg'},
                'iron': {'daily': 27, 'unit': 'mg'},
                'folic_acid': {'daily': 600, 'unit': 'mcg'},
                'vitamin_a': {'daily': 770, 'unit': 'mcg'},
                'vitamin_b6': {'daily': 1.9, 'unit': 'mg'},
                'vitamin_b12': {'daily': 2.6, 'unit': 'mcg'},
                'vitamin_c': {'daily': 85, 'unit': 'mg'},
                'vitamin_d': {'daily': 15, 'unit': 'mcg'},
                'zinc': {'daily': 11, 'unit': 'mg'},
                'magnesium': {'daily': 360, 'unit': 'mg'},
                'omega3': {'daily': 300, 'unit': 'mg'}
            },
            3: {
                'calories': {'daily': 2600, 'unit': 'kcal'},
                'protein': {'daily': 70, 'unit': 'g'},
                'carbs': {'daily': 175, 'unit': 'g'},
                'fat': {'daily': 70, 'unit': 'g'},
                'fiber': {'daily': 25, 'unit': 'g'},
                'calcium': {'daily': 1000, 'unit': 'mg'},
                'iron': {'daily': 27, 'unit': 'mg'},
                'folic_acid': {'daily': 600, 'unit': 'mcg'},
                'vitamin_a': {'daily': 770, 'unit': 'mcg'},
                'vitamin_b6': {'daily': 1.9, 'unit': 'mg'},
                'vitamin_b12': {'daily': 2.6, 'unit': 'mcg'},
                'vitamin_c': {'daily': 85, 'unit': 'mg'},
                'vitamin_d': {'daily': 15, 'unit': 'mcg'},
                'zinc': {'daily': 11, 'unit': 'mg'},
                'magnesium': {'daily': 360, 'unit': 'mg'},
                'omega3': {'daily': 300, 'unit': 'mg'}
            }
        }
        
        return targets.get(trimester, targets[1])
    
    def _generate_nutrition_summary(
        self,
        totals: Dict[str, float],
        targets: Dict[str, Dict],
        days: int,
        meal_frequency: str
    ) -> Dict:
        """Generate nutrition summary with compliance."""
        daily_avg = {key: totals[key] / days if days > 0 else 0 for key in totals}
        
        summary = {
            'total_days': days,
            'meal_frequency': meal_frequency,
            'daily_average': {},
            'compliance': {},
            'recommendations': []
        }
        
        for nutrient, avg_value in daily_avg.items():
            target_info = targets.get(nutrient, {})
            target_value = target_info.get('daily', 0)
            unit = target_info.get('unit', '')
            
            if target_value > 0:
                compliance_pct = (avg_value / target_value * 100) if target_value else 0
            else:
                compliance_pct = 100
            
            summary['daily_average'][nutrient] = {
                'value': round(avg_value, 2),
                'unit': unit,
                'target': target_value
            }
            
            summary['compliance'][nutrient] = {
                'percentage': round(compliance_pct, 1),
                'status': 'Good' if 90 <= compliance_pct <= 110 else 'Needs Attention'
            }
            
            # Add recommendations
            if compliance_pct < 90:
                summary['recommendations'].append(
                    f"Increase {nutrient} intake - currently at {compliance_pct:.0f}% of target"
                )
            elif compliance_pct > 110:
                summary['recommendations'].append(
                    f"Moderate {nutrient} intake - currently at {compliance_pct:.0f}% of target"
                )
        
        return summary
    
    def _format_meal_plan_table(self, meal_plan: List[Dict]) -> List[Dict]:
        """Format meal plan for table display."""
        formatted = []
        
        for day_plan in meal_plan:
            row = {
                'day': f"Day {day_plan['day']}",
                'date': day_plan['date'],
                'calories': 0,
                'breakfast': '-',
                'lunch': '-',
                'dinner': '-',
                'mid_morning_snack': '-',
                'evening_snack': '-',
            }
            
            total_calories = 0
            for meal_type, meal in day_plan['meals'].items():
                # Try multiple column names to extract food name
                possible_columns = ['food', 'meal', 'dish', 'food_item', 'recipe', 'name', 'item']
                meal_name = 'Unknown'
                
                for col in possible_columns:
                    if col in meal and meal[col]:
                        meal_name = str(meal[col]).strip()
                        break
                
                # Clean up the meal name
                if meal_name and meal_name != 'Unknown':
                    meal_name = meal_name.title()
                
                # Use standardized meal type names for Display
                display_type = meal_type.replace('_', ' ').title()
                row[meal_type.replace('-', '_')] = meal_name
                
                # Add calories if available
                if 'calories' in meal and isinstance(meal['calories'], (int, float)):
                    total_calories += meal['calories']
            
            row['calories'] = total_calories
            formatted.append(row)
        
        return formatted

    def _normalize_region(self, region: Optional[str]) -> Optional[str]:
        if not region:
            return None
        value = region.strip().lower()
        if 'north' in value:
            return 'North'
        if 'south' in value:
            return 'South'
        return region.strip()

    def _normalize_diet(self, diet: Optional[str]) -> Optional[str]:
        if not diet:
            return None
        value = diet.strip().lower()
        if 'veg' in value and 'non' not in value:
            return 'veg'
        if 'non' in value:
            return 'nonveg'
        if 'vegan' in value:
            return 'vegan'
        return value

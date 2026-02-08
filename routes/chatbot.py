"""Chatbot routes for AI-powered food recommendations with external API fallback."""
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime
from models import db
from models.interaction import UserInteraction

# Lazy load chatbots
_comprehensive_chatbot = None


def get_comprehensive_chatbot():
    """Get comprehensive chatbot with all datasets (lazy loading)."""
    global _comprehensive_chatbot
    if _comprehensive_chatbot is None:
        from ai_engine.comprehensive_chatbot import get_comprehensive_chatbot as load_chatbot
        _comprehensive_chatbot = load_chatbot()
    return _comprehensive_chatbot


chatbot_bp = Blueprint('chatbot', __name__)


@chatbot_bp.route('/')
@login_required
def chatbot_page():
    """Render chatbot interface page."""
    return render_template('dashboard/chatbot.html')


@chatbot_bp.route('/api/ask', methods=['POST'])
@login_required
def ask_question():
    """
    Answer user questions with FAST responses (< 3 seconds).
    Uses dataset + external AI (Solar-only) fallback.
    
    Expects JSON:
        {
            "question": "Can I eat eggs during pregnancy?",
            "trimester": 2 (optional)
        }
    
    Automatically returns Do's and Don'Ts format with fast response times.
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'success': False, 'error': 'Question is required'}), 400
        
        question = data['question'].strip()
        
        if not question or len(question) < 3:
            return jsonify({'success': False, 'error': 'Question too short (min 3 chars)'}), 400
        
        if len(question) > 500:
            return jsonify({'success': False, 'error': 'Question too long (max 500 chars)'}), 400
        
        # Get user context
        trimester = data.get('trimester')
        if trimester is None and hasattr(current_user, 'current_trimester'):
            trimester = current_user.current_trimester
        
        region = data.get('region')
        season = data.get('season')
        
        # Get comprehensive chatbot with all datasets + Solar fallback
        chatbot = get_comprehensive_chatbot()

        # Use structured answer to include dos/donts and intent metadata
        result = chatbot.answer_question_structured(
            question=question,
            trimester=trimester
        )
        # Inject region/season for logging context
        result['region'] = region
        result['season'] = season
        
        # Log interaction
        try:
            interaction = UserInteraction(
                user_id=current_user.id,
                interaction_type='chatbot_query'
            )
            interaction.set_details({
                'question': question,
                'trimester': trimester,
                'source': result.get('source'),
                'response_time': result.get('response_time'),
                'answer_length': len(result.get('answer', '')),
                'keywords': result.get('keywords', []),
                'intent': result.get('intent')
            })
            
            db.session.add(interaction)
            db.session.commit()
        except Exception as e:
            print(f"⚠️ Could not log interaction: {e}")
            db.session.rollback()
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': result.get('answer', ''),
            'dos': result.get('dos', []),
            'donts': result.get('donts', []),
            'query_reflection': result.get('query_reflection', ''),
            'keywords': result.get('keywords', []),
            'intent': result.get('intent'),
            'source': result.get('source'),  # 'dataset', 'ai_model', 'fallback'
            'response_time': round(result.get('response_time', 0), 2),
            'trimester': trimester,
            'region': region,
            'season': season,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        import traceback
        print(f"❌ Error in chatbot ask: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Error processing question',
            'answer': 'Sorry, I encountered an error. Please try again.'
        }), 500


@chatbot_bp.route('/api/suggestions', methods=['GET'])
@login_required
def get_suggestions():
    """
    Get suggested questions.
    
    Returns:
        {
            "suggestions": ["Question 1", "Question 2", ...]
        }
    """
    try:
        trimester = current_user.current_trimester if hasattr(current_user, 'current_trimester') and current_user.current_trimester else 2
        
        suggestions = [
            f"What should I eat in trimester {trimester}?",
            "What foods should I avoid during pregnancy?",
            "Can I eat eggs during pregnancy?",
            "Is fish safe during pregnancy?",
            "What are good sources of iron?",
            "Which fruits are best for pregnancy?",
            "What is a good meal plan for today?",
            "What foods help with morning sickness?"
        ]
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'trimester': trimester
        })
        
    except Exception as e:
        print(f"Error getting suggestions: {e}")
        return jsonify({
            'error': 'Could not load suggestions',
            'suggestions': []
        }), 500


@chatbot_bp.route('/api/history', methods=['GET'])
@login_required
def get_history():
    """
    Get user's chat history.
    
    Query params:
        limit: Number of recent interactions (default: 20)
    
    Returns:
        {
            "history": [
                {
                    "id": 1,
                    "question": "...",
                    "timestamp": "2024-01-01T12:00:00",
                    "source": "dataset"
                },
                ...
            ]
        }
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        limit = min(limit, 100)  # Max 100 items
        
        # Get user's chatbot interactions
        interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='chatbot_query'
        ).order_by(
            UserInteraction.timestamp.desc()
        ).limit(limit).all()
        
        history = []
        for interaction in interactions:
            details = interaction.get_details()
            history.append({
                'id': interaction.id,
                'question': details.get('question', ''),
                'intent': details.get('intent', ''),
                'foods_mentioned': details.get('foods_mentioned', []),
                'timestamp': interaction.timestamp.isoformat()
            })
        
        return jsonify({
            'success': True,
            'history': history,
            'total': len(history)
        })
        
    except Exception as e:
        print(f"Error getting history: {e}")
        return jsonify({
            'error': 'Could not load chat history',
            'history': []
        }), 500

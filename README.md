# iBaby - AI-Powered Pregnancy Nutrition App

An intelligent pregnancy nutrition application that provides personalized meal planning, nutritional guidance, and chatbot support for pregnant women, powered by advanced AI technology.

## 🌟 Features

### 1. AI-Powered Meal Planning
- **Personalized meal plans** based on trimester, region, and dietary preferences
- **Hybrid approach**: Dataset-based meals enhanced with AI nutrition validation
- **16+ nutrient tracking** including vitamins, minerals, and omega-3
- **Visual nutrition dashboard** with color-coded progress indicators
- Supports both 3-meal and 5-meal daily plans

### 2. Comprehensive Nutrition Tracking
Track 16 essential nutrients:
- **Macros**: Calories, Protein, Carbs, Fat, Fiber
- **Minerals**: Iron, Calcium, Zinc, Magnesium
- **Vitamins**: A, B6, B12, C, D, Folic Acid
- **Omega-3** fatty acids

Each nutrient shows:
- Current daily average
- Recommended target
- Compliance percentage
- Color-coded status (✅ Good, ⚠️ Needs Attention, ❌ Deficient)

### 3. AI-Enhanced Chatbot
Multi-tier intelligent chatbot with fallback logic:
1. **Dataset Search** - Answers from comprehensive pregnancy nutrition datasets
2. **Gemini AI** - Google's Gemini Pro for contextual responses (if configured)
3. **General Guidance** - Safe fallback responses

Features:
- Context-aware responses (trimester, region, diet type)
- Safety checks for food consumption
- Nutritional benefits information
- Pregnancy do's and don'ts

### 4. Regional & Dietary Support
- **Regions**: North Indian, South Indian cuisines
- **Diets**: Vegetarian, Non-Vegetarian, Vegan
- **Trimesters**: Customized nutrition for each stage
- **Special Conditions**: Diabetes, gestational diabetes support

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tejaslakshmangith/ibaby.git
   cd ibaby
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (Optional for AI features)
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the app**
   Open your browser and navigate to `http://localhost:5000`

## 🔑 Environment Variables

Create a `.env` file in the root directory with the following optional configurations:

### AI Features (Optional)
```env
# Google Gemini API (Free Tier Available)
# Get your key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your-gemini-api-key-here

# Alternative: Use GOOGLE_API_KEY instead
GOOGLE_API_KEY=your-google-api-key-here
```

### Flask Configuration
```env
FLASK_ENV=development
SECRET_KEY=your-secret-key-change-this-in-production
DATABASE_URL=sqlite:///instance/database.db
```

### AI Configuration (Advanced)
```env
# AI request timeout in seconds (default: 2.5)
AI_TIMEOUT_SECONDS=2.5

# Chatbot rate limit per minute (default: 20)
CHATBOT_RATE_LIMIT_PER_MIN=20

# Response cache TTL in seconds (default: 3600)
CACHE_TTL_SECONDS=3600
```

## 📊 Usage Examples

### Meal Planning
1. Navigate to the Meal Plans page
2. Select your preferences:
   - Number of days (1-30)
   - Region (North/South Indian)
   - Diet type (Vegetarian/Non-Vegetarian/Vegan)
   - Meals per day (3 or 5 meals)
3. Click "Generate Meal Plan"
4. View your personalized plan with comprehensive nutrition tracking

### Chatbot Queries
Ask questions like:
- "Is it safe to eat papaya during pregnancy?"
- "What foods are high in iron for pregnancy?"
- "What should I eat in the second trimester?"
- "Can I have coffee while pregnant?"

The chatbot provides:
- Safety information
- Nutritional benefits
- Trimester-specific guidance
- Medical disclaimers

## 🤖 AI Integration

### Google Gemini (Recommended - Free Tier)
1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add to `.env`: `GEMINI_API_KEY=your-key-here`
3. Restart the application

**Features enabled:**
- AI-powered chatbot responses
- Advanced nutrition calculations
- Contextual meal recommendations

**Free Tier Limits:**
- 60 requests per minute
- Rate limiting built-in
- Graceful degradation if unavailable

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/ -v

# Run AI integration tests
python tests/test_ai_integration.py

# Run with coverage
python -m pytest tests/ --cov=ai_engine --cov-report=html
```

**Test Coverage:**
- ✅ Gemini AI initialization and graceful degradation
- ✅ Meal planner AI integration
- ✅ Chatbot AI fallback logic
- ✅ Season parameter removal verification
- ✅ Enhanced nutrition tracking (16 nutrients)

## 🔒 Security

- **CodeQL Scanning**: Automated security analysis (0 vulnerabilities)
- **Graceful Degradation**: Works without AI keys
- **Rate Limiting**: Built-in protection against API abuse
- **Input Validation**: All user inputs sanitized
- **Medical Disclaimers**: Clear notices on AI-generated advice

## 📁 Project Structure

```
ibaby/
├── ai_engine/
│   ├── gemini_integration.py      # Google Gemini AI wrapper
│   ├── meal_planner.py            # Meal plan generation
│   ├── comprehensive_chatbot.py   # Multi-tier chatbot
│   └── unified_dataset_loader.py  # Dataset management
├── routes/
│   ├── meal_plans.py              # Meal planning API
│   ├── chatbot.py                 # Chatbot API
│   └── ...
├── templates/
│   └── dashboard/
│       ├── meal_plans.html        # Meal planning UI
│       └── chatbot.html           # Chatbot UI
├── data/                          # Nutrition datasets
├── tests/
│   └── test_ai_integration.py     # AI feature tests
├── .env.example                   # Environment variables template
├── requirements.txt               # Python dependencies
└── app.py                         # Flask application entry point
```

## 🔄 Recent Updates

### v2.0.0 - AI-Powered Enhancement
- ✅ Removed season dependency from meal planning
- ✅ Integrated Google Gemini AI (free tier)
- ✅ Expanded nutrition tracking to 16+ nutrients
- ✅ Visual progress bars with color-coded status
- ✅ AI-enhanced chatbot with multi-tier fallback
- ✅ Hybrid meal generation (Dataset + AI)
- ✅ Comprehensive test suite (8 tests, all passing)
- ✅ Zero security vulnerabilities (CodeQL verified)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Tejas Lakshman ([@tejaslakshmangith](https://github.com/tejaslakshmangith))

## 🙏 Acknowledgments

- Google Gemini AI for free-tier AI capabilities
- Pregnancy nutrition datasets from various open sources
- Flask framework and community

## 📧 Support

For issues, questions, or suggestions:
- 📫 Open an issue on GitHub
- 💬 Contact: [GitHub Issues](https://github.com/tejaslakshmangith/ibaby/issues)

## ⚕️ Medical Disclaimer

**Important**: This application provides nutritional information and AI-generated advice for educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.

Always consult with:
- Your doctor or healthcare provider
- A registered dietitian
- A qualified medical professional

Before making any dietary changes during pregnancy.

---

**Made with ❤️ for expecting mothers**

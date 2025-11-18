import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import pandas as pd
from torch.serialization import add_safe_globals

# Page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL CLASS DEFINITION (MUST BE INCLUDED!)
# ============================================================================
class EmailClassifier(nn.Module):
    """LSTM-based email classifier model"""
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        dropped = self.dropout(hidden[-1])
        return self.sigmoid(self.fc(dropped))
# ============================================================================

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-top: 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Preprocessing function
def preprocess_text(text):
    """Clean and tokenize text"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.split()

# Allow your custom class to be loaded safely
add_safe_globals([EmailClassifier])

# Load model and vocabulary
@st.cache_resource
def load_model_and_vocab():
    """Load trained model and vocabulary"""
    try:
        model = torch.load('email_classifier.pth', map_location=torch.device('cpu'),weights_only=False)
        model.eval()
        
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        
        return model, vocab, None
    except FileNotFoundError as e:
        return None, None, f"Model files not found: {e}"
    except Exception as e:
        return None, None, f"Error loading model: {e}"

# Prediction function
def predict(text, model, vocab, max_len=50):
    """Predict if message is spam or ham"""
    model.eval()
    tokens = preprocess_text(text)
    indices = [vocab.get(word, vocab.get('<UNK>', 1)) for word in tokens]
    
    # Pad or truncate to max_len
    if len(indices) < max_len:
        indices = indices + [vocab['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    
    input_tensor = torch.tensor([indices], dtype=torch.long)
    
    with torch.no_grad():
        prob = model(input_tensor).item()
    
    label = "Spam" if prob > 0.5 else "Ham"
    confidence = prob if prob > 0.5 else 1 - prob
    
    return label, confidence, prob

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📧 SMS Spam Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by PyTorch LSTM Neural Network</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, vocab, error = load_model_and_vocab()
    
    if error:
        st.error(f"❌ {error}")
        st.info("**Instructions:**\n1. Run the training notebook first\n2. Ensure `email_classifier.pth` and `vocab.pkl` are in the same directory")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Info")
        st.success("✅ Model Loaded")
        st.metric("Vocabulary Size", f"{len(vocab):,}")
        st.metric("Model Type", "LSTM")
        st.metric("Max Sequence Length", "50 tokens")
        
        st.markdown("---")
        
        st.header("ℹ️ About")
        st.info(
            "This classifier uses a **Long Short-Term Memory (LSTM)** neural network "
            "trained on SMS messages to detect spam."
        )
    
    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Message", "📁 Batch Upload", "🧪 Examples"])
    
    # Tab 1: Single Message Classification
    with tab1:
        st.header("Classify a Single Message")
        
        message = st.text_area(
            "Enter SMS message:",
            height=150,
            placeholder="Type or paste your message here..."
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            classify_btn = st.button("🔍 Classify Message", type="primary", use_container_width=True)
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.rerun()
        
        if classify_btn and message.strip():
            with st.spinner("🔄 Analyzing message..."):
                label, confidence, raw_prob = predict(message, model, vocab)
            
            st.markdown("### 🎯 Classification Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if label == "Spam":
                    st.error(f"### 🚫 {label}")
                else:
                    st.success(f"### ✅ {label}")
            
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col3:
                st.metric("Spam Score", f"{raw_prob:.4f}")
            
            st.progress(raw_prob, text=f"Spam probability: {raw_prob:.1%}")
            
            if label == "Spam":
                st.warning("⚠️ **Warning:** This message appears to be spam.")
            else:
                st.info("✓ **Legitimate:** This message appears to be genuine.")
    
    # Tab 2: Batch Upload
    with tab2:
        st.header("Batch Message Classification")
        
        uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
        
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            messages = [line.strip() for line in content.split('\n') if line.strip()]
            
            st.info(f"📊 Found **{len(messages)}** messages")
            
            if st.button("🔍 Classify All", type="primary"):
                results = []
                progress_bar = st.progress(0)
                
                for i, msg in enumerate(messages):
                    label, conf, prob = predict(msg, model, vocab)
                    results.append({
                        'Message': msg[:60] + '...' if len(msg) > 60 else msg,
                        'Classification': label,
                        'Confidence': f"{conf:.1%}",
                        'Spam Score': f"{prob:.4f}"
                    })
                    progress_bar.progress((i + 1) / len(messages))
                
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                spam_count = sum(1 for r in results if r['Classification'] == 'Spam')
                ham_count = len(results) - spam_count
                
                col1, col2 = st.columns(2)
                col1.metric("🚫 Spam", spam_count)
                col2.metric("✅ Ham", ham_count)
                
                csv = df_results.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "results.csv",
                    "text/csv"
                )
    
    # Tab 3: Examples
    with tab3:
        st.header("🧪 Try Example Messages")
        
        examples = [
            ("Spam", "FREE! Win £1000 now! Call immediately!"),
            ("Ham", "Hey, coffee tomorrow at 3pm?"),
            ("Spam", "URGENT: Verify your account now!"),
            ("Ham", "Can you send the report please?"),
            ("Spam", "Congratulations! You won an iPhone!"),
            ("Ham", "Thanks for dinner last night!"),
        ]
        
        for i, (expected, example) in enumerate(examples):
            with st.expander(f"{'🚫' if expected == 'Spam' else '✅'} {expected}: {example[:40]}..."):
                st.write(f"**Full message:** {example}")
                
                if st.button(f"Classify", key=f"ex_{i}"):
                    label, conf, prob = predict(example, model, vocab)
                    
                    if label == "Spam":
                        st.error(f"**Result:** 🚫 {label}")
                    else:
                        st.success(f"**Result:** ✅ {label}")
                    
                    st.metric("Confidence", f"{conf:.1%}")
                    st.progress(prob)
                    
                    if label == expected:
                        st.success("✅ Correct prediction!")
                    else:
                        st.error("❌ Incorrect prediction")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with PyTorch and Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

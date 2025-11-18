# import streamlit as st
# import torch
# import pickle
# import re
# import pandas as pd

# # Page configuration
# st.set_page_config(
#     page_title="SMS Spam Classifier",
#     page_icon="📧",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 0;
#     }
#     .sub-header {
#         text-align: center;
#         color: #666;
#         margin-top: 0;
#     }
#     .stButton>button {
#         width: 100%;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Preprocessing function
# def preprocess_text(text):
#     """Clean and tokenize text"""
#     text = str(text).lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     return text.split()

# # Load model and vocabulary
# @st.cache_resource
# def load_model_and_vocab():
#     """Load trained model and vocabulary"""
#     try:
#         model = torch.load('email_classifier.pth', map_location=torch.device('cpu'))
#         model.eval()
        
#         with open('vocab.pkl', 'rb') as f:
#             vocab = pickle.load(f)
        
#         return model, vocab, None
#     except FileNotFoundError as e:
#         return None, None, f"Model files not found: {e}"
#     except Exception as e:
#         return None, None, f"Error loading model: {e}"

# # Prediction function
# def predict(text, model, vocab, max_len=50):
#     """Predict if message is spam or ham"""
#     model.eval()
#     tokens = preprocess_text(text)
#     indices = [vocab.get(word, vocab.get('<UNK>', 1)) for word in tokens]
    
#     # Pad or truncate to max_len
#     if len(indices) < max_len:
#         indices = indices + [vocab['<PAD>']] * (max_len - len(indices))
#     else:
#         indices = indices[:max_len]
    
#     input_tensor = torch.tensor([indices], dtype=torch.long)
    
#     with torch.no_grad():
#         prob = model(input_tensor).item()
    
#     label = "Spam" if prob > 0.5 else "Ham"
#     confidence = prob if prob > 0.5 else 1 - prob
    
#     return label, confidence, prob

# # Main app
# def main():
#     # Header
#     st.markdown('<h1 class="main-header">📧 SMS Spam Classifier</h1>', unsafe_allow_html=True)
#     st.markdown('<p class="sub-header">Powered by PyTorch LSTM Neural Network</p>', unsafe_allow_html=True)
#     st.markdown("---")
    
#     # Load model
#     model, vocab, error = load_model_and_vocab()
    
#     if error:
#         st.error(f"❌ {error}")
#         st.info("**Instructions:**\n1. Run the training notebook first\n2. Ensure `email_classifier.pth` and `vocab.pkl` are in the same directory")
#         st.stop()
    
#     # Sidebar
#     with st.sidebar:
#         st.header("📊 Model Info")
#         st.success("✅ Model Loaded")
#         st.metric("Vocabulary Size", f"{len(vocab):,}")
#         st.metric("Model Type", "LSTM")
#         st.metric("Max Sequence Length", "50 tokens")
        
#         st.markdown("---")
        
#         st.header("ℹ️ About")
#         st.info(
#             "This classifier uses a **Long Short-Term Memory (LSTM)** neural network "
#             "trained on SMS messages to detect spam.\n\n"
#             "**Features:**\n"
#             "- Real-time classification\n"
#             "- Confidence scores\n"
#             "- Batch processing\n"
#             "- Example messages"
#         )
        
#         st.markdown("---")
        
#         st.header("📈 How it works")
#         st.write(
#             "1. **Text preprocessing**: Lowercasing, removing special characters\n"
#             "2. **Tokenization**: Converting words to vocabulary indices\n"
#             "3. **LSTM encoding**: Capturing sequential patterns\n"
#             "4. **Classification**: Spam probability prediction"
#         )
    
#     # Main content - Tabs
#     tab1, tab2, tab3, tab4 = st.tabs(["📝 Single Message", "📁 Batch Upload", "🧪 Try Examples", "📊 Statistics"])
    
#     # Tab 1: Single Message Classification
#     with tab1:
#         st.header("Classify a Single Message")
        
#         col1, col2 = st.columns([3, 1])
        
#         with col1:
#             message = st.text_area(
#                 "Enter SMS message:",
#                 height=150,
#                 placeholder="Type or paste your message here...",
#                 help="Enter the text message you want to classify"
#             )
        
#         with col2:
#             st.write("**Quick Actions:**")
#             if st.button("🗑️ Clear", use_container_width=True):
#                 st.rerun()
            
#             classify_btn = st.button("🔍 Classify Message", type="primary", use_container_width=True)
        
#         if classify_btn:
#             if message.strip():
#                 with st.spinner("🔄 Analyzing message..."):
#                     label, confidence, raw_prob = predict(message, model, vocab)
                
#                 st.markdown("### 🎯 Classification Result")
                
#                 # Result display
#                 col1, col2, col3 = st.columns(3)
                
#                 with col1:
#                     if label == "Spam":
#                         st.error(f"### 🚫 {label}")
#                     else:
#                         st.success(f"### ✅ {label}")
                
#                 with col2:
#                     st.metric("Confidence", f"{confidence:.1%}", help="How confident the model is in its prediction")
                
#                 with col3:
#                     st.metric("Spam Score", f"{raw_prob:.4f}", help="Raw probability of being spam (0-1)")
                
#                 # Progress bar
#                 st.progress(raw_prob, text=f"Spam probability: {raw_prob:.1%}")
                
#                 # Recommendation
#                 if label == "Spam":
#                     st.warning("⚠️ **Warning:** This message appears to be spam. Be cautious of:")
#                     st.markdown("- Requests for personal information\n- Suspicious links\n- Urgent calls to action\n- Prize/money offers")
#                 else:
#                     st.info("✓ **Legitimate:** This message appears to be a genuine communication.")
                
#                 # Message details
#                 with st.expander("📋 Message Details"):
#                     st.write(f"**Original message:**\n> {message}")
#                     st.write(f"**Length:** {len(message)} characters")
#                     st.write(f"**Word count:** {len(message.split())} words")
#                     tokens = preprocess_text(message)
#                     st.write(f"**Tokens after processing:** {len(tokens)}")
#             else:
#                 st.warning("⚠️ Please enter a message to classify.")
    
#     # Tab 2: Batch Upload
#     with tab2:
#         st.header("Batch Message Classification")
#         st.write("Upload a text file with one message per line")
        
#         uploaded_file = st.file_uploader(
#             "Choose a text file",
#             type=['txt'],
#             help="Upload a .txt file with one message per line"
#         )
        
#         if uploaded_file:
#             content = uploaded_file.read().decode('utf-8')
#             messages = [line.strip() for line in content.split('\n') if line.strip()]
            
#             st.info(f"📊 Found **{len(messages)}** messages in file")
            
#             # Show preview
#             with st.expander("👁️ Preview first 5 messages"):
#                 for i, msg in enumerate(messages[:5], 1):
#                     st.write(f"{i}. {msg[:100]}{'...' if len(msg) > 100 else ''}")
            
#             if st.button("🔍 Classify All Messages", type="primary"):
#                 results = []
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()
                
#                 for i, msg in enumerate(messages):
#                     status_text.text(f"Processing message {i+1}/{len(messages)}...")
#                     label, conf, prob = predict(msg, model, vocab)
#                     results.append({
#                         'Message': msg[:80] + '...' if len(msg) > 80 else msg,
#                         'Classification': label,
#                         'Confidence': f"{conf:.1%}",
#                         'Spam Score': f"{prob:.4f}"
#                     })
#                     progress_bar.progress((i + 1) / len(messages))
                
#                 status_text.text("✅ Classification complete!")
                
#                 # Display results
#                 st.markdown("### 📊 Classification Results")
#                 df_results = pd.DataFrame(results)
#                 st.dataframe(df_results, use_container_width=True, height=400)
                
#                 # Summary statistics
#                 st.markdown("### 📈 Summary Statistics")
#                 spam_count = sum(1 for r in results if r['Classification'] == 'Spam')
#                 ham_count = len(results) - spam_count
                
#                 col1, col2, col3, col4 = st.columns(4)
#                 col1.metric("Total Messages", len(results))
#                 col2.metric("🚫 Spam", spam_count)
#                 col3.metric("✅ Ham", ham_count)
#                 col4.metric("Spam Rate", f"{spam_count/len(results)*100:.1f}%")
                
#                 # Download results
#                 csv = df_results.to_csv(index=False)
#                 st.download_button(
#                     label="📥 Download Results as CSV",
#                     data=csv,
#                     file_name="spam_classification_results.csv",
#                     mime="text/csv"
#                 )
    
#     # Tab 3: Try Examples
#     with tab3:
#         st.header("🧪 Try Example Messages")
#         st.write("Click on any example to see how the model classifies it")
        
#         examples = [
#             ("Spam", "WINNER!! You have been selected to receive a £1000 cash prize! Call now to claim your reward!"),
#             ("Ham", "Hey, are you free for coffee tomorrow afternoon? Let me know what time works for you."),
#             ("Spam", "FREE entry in 2 a weekly comp to win FA Cup final tickets. Text FA to 87121 to receive entry question"),
#             ("Ham", "Can you please send me the report by end of day? Thanks for your help on this project."),
#             ("Spam", "URGENT! Your bank account has been compromised. Click here immediately to verify your identity."),
#             ("Ham", "Don't forget about the meeting at 3pm in conference room B. See you there!"),
#             ("Spam", "Congratulations! You've won a FREE iPhone 15 Pro! Click this link now to claim your prize before it expires!"),
#             ("Ham", "Thanks for dinner last night. We should do it again soon. How about next Friday?"),
#             ("Spam", "CALL NOW! Limited time offer - lose 20 pounds in 2 weeks! 100% guaranteed or money back!"),
#             ("Ham", "The package arrived safely. Thank you for shipping it so quickly. Really appreciate it."),
#         ]
        
#         for i, (expected, example) in enumerate(examples):
#             with st.expander(f"{'🚫 SPAM' if expected == 'Spam' else '✅ HAM'}: {example[:60]}..."):
#                 st.write(f"**Expected:** {expected}")
#                 st.write(f"**Full message:** {example}")
                
#                 if st.button(f"Classify this message", key=f"example_{i}"):
#                     label, conf, prob = predict(example, model, vocab)
                    
#                     col1, col2 = st.columns(2)
                    
#                     with col1:
#                         if label == "Spam":
#                             st.error(f"**Prediction:** 🚫 {label}")
#                         else:
#                             st.success(f"**Prediction:** ✅ {label}")
                    
#                     with col2:
#                         st.metric("Confidence", f"{conf:.1%}")
                    
#                     st.progress(prob)
                    
#                     # Check if correct
#                     if label == expected:
#                         st.success("✅ **Correct!** The model predicted the right class.")
#                     else:
#                         st.error("❌ **Incorrect.** The model made a mistake here.")
    
#     # Tab 4: Statistics
#     with tab4:
#         st.header("📊 Model Statistics")
        
#         st.markdown("### 🎯 Performance Metrics")
        
#         # Mock statistics (replace with actual if you track them)
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.metric("Training Accuracy", "96.7%", help="Accuracy on training dataset")
        
#         with col2:
#             st.metric("Test Accuracy", "95.3%", help="Accuracy on test dataset")
        
#         with col3:
#             st.metric("Spam Recall", "94.1%", help="Percentage of spam correctly identified")
        
#         st.markdown("---")
        
#         st.markdown("### 📈 Model Architecture")
        
#         architecture_info = """
#         **Layer Structure:**
#         - **Embedding Layer**: 100 dimensions
#         - **LSTM Layer**: 100 hidden units
#         - **Dropout**: 50% (prevents overfitting)
#         - **Dense Layer**: Binary classification
#         - **Activation**: Sigmoid
        
#         **Training Configuration:**
#         - **Optimizer**: Adam
#         - **Learning Rate**: 0.0005
#         - **Batch Size**: 16
#         - **Epochs**: 30
#         - **Loss Function**: Binary Cross-Entropy
#         """
        
#         st.markdown(architecture_info)
        
#         st.markdown("---")
        
#         st.markdown("### 🔍 Common Spam Indicators")
        
#         spam_indicators = """
#         The model has learned to identify these common spam patterns:
        
#         1. **Urgent language**: "URGENT", "ACT NOW", "LIMITED TIME"
#         2. **Free offers**: "FREE", "WIN", "PRIZE"
#         3. **Financial terms**: "CASH", "£££", "$$$", "MONEY"
#         4. **Call to action**: "CALL NOW", "CLICK HERE", "TEXT TO"
#         5. **Account warnings**: "VERIFY", "SUSPENDED", "COMPROMISED"
#         6. **Excessive punctuation**: "!!!", "???", "..."
#         """
#         st.markdown(spam_indicators)
#     # Footer
#     st.markdown("---")
#     st.markdown(
#         "<div style='text-align: center; color: #666;'>"
#         "Built with ❤️ using PyTorch and Streamlit | "
#         "🔒 Your messages are processed locally and not stored"
#         "</div>",
#         unsafe_allow_html=True
#     )
# if __name__ == "__main__":
#     main()

import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import pandas as pd

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

# Load model and vocabulary
@st.cache_resource
def load_model_and_vocab():
    """Load trained model and vocabulary"""
    try:
        model = torch.load('email_classifier.pth', map_location=torch.device('cpu'))
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

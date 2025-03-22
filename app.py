from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import shap
import matplotlib.pyplot as plt
import io
import base64
import glob

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['UPLOAD_FOLDER'] = r'C:\Dhruv\plant_disease_flask_app\static\uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

# Post model
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(300), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    votes = db.Column(db.Integer, default=0)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('posts', lazy=True))
    votes_relationship = db.relationship('PostVote', backref='post', lazy=True)

# Comment model
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    votes = db.Column(db.Integer, default=0)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    post = db.relationship('Post', backref=db.backref('comments', lazy=True))
    user = db.relationship('User', backref=db.backref('comments', lazy=True))
    votes_relationship = db.relationship('CommentVote', backref='comment', lazy=True)

# Post Vote model
class PostVote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    vote_type = db.Column(db.Integer, nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'post_id', name='unique_user_post_vote'),)

# Comment Vote model
class CommentVote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    comment_id = db.Column(db.Integer, db.ForeignKey('comment.id'), nullable=False)
    vote_type = db.Column(db.Integer, nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'comment_id', name='unique_user_comment_vote'),)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

# Load TensorFlow model and initialize SHAP
try:
    model = tf.keras.models.load_model(r"C:\Dhruv\Streamlit_Dashboard\trained_plant_disease_model_5.h5")
    print("Model loaded successfully")
    image_paths = glob.glob(r"C:\Dhruv\plant_disease_flask_app\test\*.jpg")[:30]
    if not image_paths:
        raise Exception("No test images found in C:\Dhruv\plant_disease_flask_app\test")
    print(f"Found {len(image_paths)} test images: {image_paths}")
    background_data = np.array([tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(p, target_size=(128, 128))) / 255.0 for p in image_paths])
    print(f"Background data shape: {background_data.shape}")
    explainer = shap.DeepExplainer(model, background_data)
    print("SHAP DeepExplainer initialized successfully")
except Exception as e:
    print(f"Error loading model or initializing SHAP: {e}")
    model = None
    explainer = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def model_prediction(image_path):
    if model is None:
        raise Exception("Model not loaded properly")
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    input_arr = np.array([input_arr])
    print(f"Input array shape: {input_arr.shape}")
    predictions = model.predict(input_arr)
    print(f"Predictions shape: {predictions.shape}, Predicted class: {np.argmax(predictions)}")
    return np.argmax(predictions), input_arr

def get_shap_explanation(input_arr, predicted_class):
    if explainer is None:
        print("SHAP explainer is None - not initialized")
        return None
    try:
        print(f"Generating SHAP values for class {predicted_class}")
        shap_values = explainer.shap_values(input_arr)
        print(f"SHAP values shape: {np.array(shap_values).shape}")
        # Correct indexing: batch index 0, class at last axis
        shap_values_for_class = shap_values[0, :, :, :, predicted_class]
        print(f"SHAP values for class shape: {shap_values_for_class.shape}")
        
        plt.figure()
        shap.image_plot(shap_values_for_class, input_arr[0], show=False)
        plt.savefig('shap_plot.png', bbox_inches='tight')
        print("SHAP plot saved to shap_plot.png")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        print("SHAP image generated successfully")
        return img_base64
    except Exception as e:
        print(f"SHAP explanation error: {e}")
        return None

def format_class_name(class_name):
    parts = class_name.replace('___', '_').split('_')
    parts = [part for part in parts if part]
    if 'healthy' in parts[-1].lower():
        return f"Healthy {parts[0]}"
    plant = parts[0].replace('_(maize)', ' (Maize)')
    disease = ' '.join(parts[1:]).replace('_', ' ')
    return f"{plant} {disease}"

class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
              'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
              'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
              'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
              'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
              'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
              'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
              'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
              'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
              'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
              'Tomato___healthy']

treatments = {
    'Apple___Apple_scab': 'Apply fungicides like captan or myclobutanil. Remove and destroy affected leaves.',
    'Apple___Black_rot': 'Prune infected branches, apply fungicides such as sulfur or copper-based products.',
    'Apple___Cedar_apple_rust': 'Use fungicides like triadimefon, remove nearby cedar trees if possible.',
    'Apple___healthy': 'No treatment needed. Maintain regular watering and fertilization.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Apply foliar fungicides like azoxystrobin, improve air circulation.',
    'Corn_(maize)___Common_rust_': 'Use resistant varieties, apply fungicides like mancozeb.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Rotate crops, apply fungicides such as chlorothalonil.',
    'Corn_(maize)___healthy': 'No treatment required. Ensure proper nutrient levels.',
    'Grape___Black_rot': 'Apply fungicides like myclobutanil, remove infected berries and leaves.',
    'Grape___Esca_(Black_Measles)': 'Prune affected vines, no effective chemical treatment; focus on prevention.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Use copper-based fungicides, improve vineyard sanitation.',
    'Grape___healthy': 'Maintain vine health with balanced fertilization and pruning.',
    'Potato___Early_blight': 'Apply fungicides like chlorothalonil, remove infected foliage.',
    'Potato___Late_blight': 'Use fungicides like metalaxyl, destroy infected plants immediately.',
    'Potato___healthy': 'No treatment needed. Monitor for pests and diseases.',
    'Tomato___Bacterial_spot': 'Apply copper-based bactericides, remove and destroy affected parts.',
    'Tomato___Early_blight': 'Use fungicides like mancozeb, rotate crops.',
    'Tomato___Late_blight': 'Apply fungicides like chlorothalonil, avoid overhead watering.',
    'Tomato___Leaf_Mold': 'Improve ventilation, apply fungicides like copper hydroxide.',
    'Tomato___Septoria_leaf_spot': 'Use fungicides like azoxystrobin, remove lower leaves.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Apply miticides like abamectin, increase humidity.',
    'Tomato___Target_Spot': 'Use fungicides like chlorothalonil, remove affected leaves.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whiteflies with insecticides, remove infected plants.',
    'Tomato___Tomato_mosaic_virus': 'Disinfect tools, remove and destroy infected plants.',
    'Tomato___healthy': 'No treatment needed. Maintain proper spacing and nutrition.'
}

symptoms = {
    'Apple___Apple_scab': 'Dark olive-green spots on leaves with velvety texture, later turning brown and cracked.',
    'Apple___Black_rot': 'Small brown spots on leaves and fruit, expanding into larger rotted areas with black pycnidia.',
    'Apple___Cedar_apple_rust': 'Yellow-orange spots on leaves with small tube-like structures, leading to leaf drop.',
    'Apple___healthy': 'No visible symptoms. Leaves are green and uniformly healthy.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Grayish-white spots with brown margins on leaves, often merging.',
    'Corn_(maize)___Common_rust_': 'Pustules of reddish-brown rust spores on both leaf surfaces, leading to drying.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Long, cigar-shaped gray-green lesions on leaves, turning tan as disease progresses.',
    'Corn_(maize)___healthy': 'No symptoms. Leaves are green and robust with no spots or lesions.',
    'Grape___Black_rot': 'Small brown spots on leaves and berries, expanding into larger rotted areas with black fungal growth.',
    'Grape___Esca_(Black_Measles)': 'Tiger-striped discoloration on leaves, black margins, and fruit rot.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Dark brown to black spots with yellow halos on leaves.',
    'Grape___healthy': 'No symptoms. Leaves are green and vines are vigorous.',
    'Potato___Early_blight': 'Dark spots on leaves with circles around them and yellow rings. Leaves dry out and fall off.',
    'Potato___Late_blight': 'Water-soaked spots on leaves turning brown with white fungal growth on undersides.',
    'Potato___healthy': 'No symptoms. Plants show uniform green foliage and healthy tubers.',
    'Tomato___Bacterial_spot': 'Small, dark brown to black spots on leaves and fruit, often with a yellow halo.',
    'Tomato___Early_blight': 'Dark spots on leaves with circles around them and yellow rings. Leaves dry out and fall off.',
    'Tomato___Late_blight': 'Water-soaked spots on leaves and stems turning brown, with white mold on undersides.',
    'Tomato___Leaf_Mold': 'Yellowing on upper leaf surfaces with olive-green to brown mold on undersides.',
    'Tomato___Septoria_leaf_spot': 'Small gray spots with dark borders on lower leaves, spreading upward.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tiny yellow or white spots on leaves, webbing, and leaf stippling.',
    'Tomato___Target_Spot': 'Dark brown spots with concentric rings on leaves, resembling a target.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Upward leaf curling, yellowing, and stunted growth.',
    'Tomato___Tomato_mosaic_virus': 'Mottled yellow and green patterns on leaves, stunted growth.',
    'Tomato___healthy': 'No symptoms. Plants show uniform green leaves and healthy fruit.'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/community', methods=['GET', 'POST'])
def community():
    if request.method == 'POST' and current_user.is_authenticated:
        title = request.form['title']
        content = request.form['content']
        image = request.files.get('image')
        image_url = None
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            image_url = url_for('static', filename='uploads/' + filename)
        post = Post(title=title, content=content, image_url=image_url, user_id=current_user.id)
        db.session.add(post)
        db.session.commit()
        flash('Post created successfully!', 'success')
        return redirect(url_for('community'))
    
    posts = Post.query.order_by(Post.date.desc()).all()
    for post in posts:
        post.voted_by_current_user = False
        if current_user.is_authenticated:
            post.voted_by_current_user = PostVote.query.filter_by(
                user_id=current_user.id, post_id=post.id).first() is not None
    return render_template('community.html', posts=posts)

@app.route('/post/<int:post_id>', methods=['GET', 'POST'])
def post_detail(post_id):
    post = Post.query.get_or_404(post_id)
    if request.method == 'POST' and current_user.is_authenticated:
        content = request.form['content']
        comment = Comment(content=content, post_id=post.id, user_id=current_user.id)
        db.session.add(comment)
        db.session.commit()
        flash('Comment added successfully!', 'success')
        return redirect(url_for('post_detail', post_id=post.id))
    post.voted_by_current_user = False
    if current_user.is_authenticated:
        post.voted_by_current_user = PostVote.query.filter_by(
            user_id=current_user.id, post_id=post.id).first() is not None
    return render_template('post_detail.html', post=post)

@app.route('/post/<int:post_id>/upvote')
@login_required
def upvote_post(post_id):
    post = Post.query.get_or_404(post_id)
    existing_vote = PostVote.query.filter_by(user_id=current_user.id, post_id=post_id).first()
    if not existing_vote:
        vote = PostVote(user_id=current_user.id, post_id=post_id, vote_type=1)
        post.votes += 1
        db.session.add(vote)
        db.session.commit()
        flash('Post upvoted!', 'success')
    else:
        flash('You have already voted on this post!', 'warning')
    return redirect(request.referrer or url_for('community'))

@app.route('/post/<int:post_id>/downvote')
@login_required
def downvote_post(post_id):
    post = Post.query.get_or_404(post_id)
    existing_vote = PostVote.query.filter_by(user_id=current_user.id, post_id=post_id).first()
    if not existing_vote:
        vote = PostVote(user_id=current_user.id, post_id=post_id, vote_type=-1)
        post.votes -= 1
        db.session.add(vote)
        db.session.commit()
        flash('Post downvoted!', 'success')
    else:
        flash('You have already voted on this post!', 'warning')
    return redirect(request.referrer or url_for('community'))

@app.route('/post/<int:post_id>/delete')
@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.user_id != current_user.id:
        flash('You can only delete your own posts!', 'error')
        return redirect(url_for('community'))
    
    PostVote.query.filter_by(post_id=post_id).delete()
    Comment.query.filter_by(post_id=post_id).delete()
    
    if post.image_url:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(post.image_url))
        if os.path.exists(image_path):
            os.remove(image_path)
    db.session.delete(post)
    db.session.commit()
    flash('Post deleted successfully!', 'success')
    return redirect(url_for('community'))

@app.route('/comment/<int:comment_id>/upvote')
@login_required
def upvote_comment(comment_id):
    comment = Comment.query.get_or_404(comment_id)
    existing_vote = CommentVote.query.filter_by(user_id=current_user.id, comment_id=comment_id).first()
    if not existing_vote:
        vote = CommentVote(user_id=current_user.id, comment_id=comment_id, vote_type=1)
        comment.votes += 1
        db.session.add(vote)
        db.session.commit()
        flash('Comment upvoted!', 'success')
    else:
        flash('You have already voted on this comment!', 'warning')
    return redirect(request.referrer or url_for('post_detail', post_id=comment.post_id))

@app.route('/comment/<int:comment_id>/downvote')
@login_required
def downvote_comment(comment_id):
    comment = Comment.query.get_or_404(comment_id)
    existing_vote = CommentVote.query.filter_by(user_id=current_user.id, comment_id=comment_id).first()
    if not existing_vote:
        vote = CommentVote(user_id=current_user.id, comment_id=comment_id, vote_type=-1)
        comment.votes -= 1
        db.session.add(vote)
        db.session.commit()
        flash('Comment downvoted!', 'success')
    else:
        flash('You have already voted on this comment!', 'warning')
    return redirect(request.referrer or url_for('post_detail', post_id=comment.post_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
        else:
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Registered successfully! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                result_index, input_arr = model_prediction(file_path)
                raw_class = class_name[result_index]
                predicted_class = format_class_name(raw_class)
                treatment = treatments.get(raw_class, "No treatment information available.")
                symptom = symptoms.get(raw_class, "No symptom information available.")
                image_url = url_for('static', filename='uploads/' + filename)

                if 'share_to_community' in request.form:
                    title = f"Prediction: {predicted_class}"
                    content = f"I got this prediction: {predicted_class}\nSymptoms: {symptom}\nTreatment: {treatment}"
                    post = Post(title=title, content=content, image_url=image_url, user_id=current_user.id)
                    db.session.add(post)
                    db.session.commit()
                    flash('Prediction shared to community!', 'success')
                    return redirect(url_for('community'))

                return render_template('predict.html', image_url=image_url, prediction=predicted_class, 
                                     treatment=treatment, symptom=symptom, filename=filename)
            except Exception as e:
                flash(f'Prediction error: {str(e)}', 'error')
                return redirect(request.url)

    return render_template('predict.html')

@app.route('/shap_explanation/<filename>')
@login_required
def shap_explanation(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        flash('Image not found.', 'error')
        return redirect(url_for('predict'))
    
    try:
        result_index, input_arr = model_prediction(file_path)
        shap_image = get_shap_explanation(input_arr, result_index)
        if shap_image:
            return render_template('shap_explanation.html', shap_image=shap_image, filename=filename)
        else:
            flash('Failed to generate SHAP explanation.', 'error')
            return redirect(url_for('predict'))
    except Exception as e:
        flash(f'SHAP error: {str(e)}', 'error')
        return redirect(url_for('predict'))

if __name__ == '__main__':
    app.run(debug=True)
# from flask import Flask, render_template, request, redirect, url_for, flash
# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
# from flask_sqlalchemy import SQLAlchemy
# from werkzeug.security import generate_password_hash, check_password_hash
# import tensorflow as tf
# import numpy as np
# import os
# from werkzeug.utils import secure_filename
# from datetime import datetime
# import shap
# import matplotlib.pyplot as plt
# import io
# import base64

# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your-secret-key'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
# app.config['UPLOAD_FOLDER'] = r'C:\Dhruv\plant_disease_flask_app\static\uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# db = SQLAlchemy(app)
# login_manager = LoginManager(app)
# login_manager.login_view = 'login'

# # User, Post, Comment, PostVote, CommentVote models (unchanged)
# class User(UserMixin, db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(150), unique=True, nullable=False)
#     password_hash = db.Column(db.String(150), nullable=False)
#     def set_password(self, password):
#         self.password_hash = generate_password_hash(password)
#     def check_password(self, password):
#         return check_password_hash(self.password_hash, password)
#     def __repr__(self):
#         return f'<User {self.username}>'

# class Post(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     title = db.Column(db.String(200), nullable=False)
#     content = db.Column(db.Text, nullable=False)
#     image_url = db.Column(db.String(300), nullable=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     votes = db.Column(db.Integer, default=0)
#     date = db.Column(db.DateTime, default=datetime.utcnow)
#     user = db.relationship('User', backref=db.backref('posts', lazy=True))
#     votes_relationship = db.relationship('PostVote', backref='post', lazy=True)

# class Comment(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     content = db.Column(db.Text, nullable=False)
#     post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     votes = db.Column(db.Integer, default=0)
#     date = db.Column(db.DateTime, default=datetime.utcnow)
#     post = db.relationship('Post', backref=db.backref('comments', lazy=True))
#     user = db.relationship('User', backref=db.backref('comments', lazy=True))
#     votes_relationship = db.relationship('CommentVote', backref='comment', lazy=True)

# class PostVote(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
#     vote_type = db.Column(db.Integer, nullable=False)
#     __table_args__ = (db.UniqueConstraint('user_id', 'post_id', name='unique_user_post_vote'),)

# class CommentVote(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     comment_id = db.Column(db.Integer, db.ForeignKey('comment.id'), nullable=False)
#     vote_type = db.Column(db.Integer, nullable=False)
#     __table_args__ = (db.UniqueConstraint('user_id', 'comment_id', name='unique_user_comment_vote'),)

# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))

# with app.app_context():
#     db.create_all()

# try:
#     model = tf.keras.models.load_model(r"C:\Dhruv\Streamlit_Dashboard\trained_plant_disease_model_5.h5")
#     # Load 10 test images for SHAP background
#     import glob
#     image_paths = glob.glob(r"C:\Dhruv\plant_disease_flask_app\test\*.jpg")[:10]
#     if not image_paths:
#         raise Exception("not wo")
#     print(f"Found {len(image_paths)} test images: {image_paths}")
#     background_data = np.array([tf.keras.preprocessing.image.img_to_array(
#         tf.keras.preprocessing.image.load_img(p, target_size=(128, 128))) for p in image_paths])
#     print(f"Background data shape: {background_data.shape}")
#     explainer = shap.DeepExplainer(model, background_data)
# except Exception as e:
#     print(f"Error loading model or initializing SHAP: {e}")
#     model = None
#     explainer = None
    
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def model_prediction(image_path):
#     if model is None:
#         raise Exception("Model not loaded properly")
#     image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = np.array([input_arr])
#     predictions = model.predict(input_arr)
#     return np.argmax(predictions), input_arr

# def get_shap_explanation(input_arr, predicted_class):
#     if explainer is None:
#         print("SHAP explainer is None - not initialized")
#         return None
#     try:
#         print(f"Generating SHAP values for class {predicted_class}")
#         shap_values = explainer.shap_values(input_arr)
#         print(f"SHAP values shape: {np.array(shap_values).shape}")
#         shap_values_for_class = shap_values[predicted_class][0]
        
#         plt.figure()
#         shap.image_plot(shap_values_for_class, input_arr[0], show=False)
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png', bbox_inches='tight')
#         buf.seek(0)
#         img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#         plt.close()
#         print("SHAP image generated successfully")
#         return img_base64
#     except Exception as e:
#         print(f"SHAP explanation error: {e}")
#         return None

# def format_class_name(class_name):
#     parts = class_name.replace('___', '_').split('_')
#     parts = [part for part in parts if part]
#     if 'healthy' in parts[-1].lower():
#         return f"Healthy {parts[0]}"
#     plant = parts[0].replace('_(maize)', ' (Maize)')
#     disease = ' '.join(parts[1:]).replace('_', ' ')
#     return f"{plant} {disease}"
# # Class labels
# class_name = [
#     'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
#     'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
#     'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
#     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
#     'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
#     'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
#     'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
#     'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
#     'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
#     'Tomato___healthy'
# ]

# # Treatments dictionary
# treatments = {
#     'Apple___Apple_scab': 'Apply fungicides like captan or myclobutanil. Remove and destroy affected leaves.',
#     'Apple___Black_rot': 'Prune infected branches, apply fungicides such as sulfur or copper-based products.',
#     'Apple___Cedar_apple_rust': 'Use fungicides like triadimefon, remove nearby cedar trees if possible.',
#     'Apple___healthy': 'No treatment needed. Maintain regular watering and fertilization.',
#     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Apply foliar fungicides like azoxystrobin, improve air circulation.',
#     'Corn_(maize)___Common_rust_': 'Use resistant varieties, apply fungicides like mancozeb.',
#     'Corn_(maize)___Northern_Leaf_Blight': 'Rotate crops, apply fungicides such as chlorothalonil.',
#     'Corn_(maize)___healthy': 'No treatment required. Ensure proper nutrient levels.',
#     'Grape___Black_rot': 'Apply fungicides like myclobutanil, remove infected berries and leaves.',
#     'Grape___Esca_(Black_Measles)': 'Prune affected vines, no effective chemical treatment; focus on prevention.',
#     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Use copper-based fungicides, improve vineyard sanitation.',
#     'Grape___healthy': 'Maintain vine health with balanced fertilization and pruning.',
#     'Potato___Early_blight': 'Apply fungicides like chlorothalonil, remove infected foliage.',
#     'Potato___Late_blight': 'Use fungicides like metalaxyl, destroy infected plants immediately.',
#     'Potato___healthy': 'No treatment needed. Monitor for pests and diseases.',
#     'Tomato___Bacterial_spot': 'Apply copper-based bactericides, remove and destroy affected parts.',
#     'Tomato___Early_blight': 'Use fungicides like mancozeb, rotate crops.',
#     'Tomato___Late_blight': 'Apply fungicides like chlorothalonil, avoid overhead watering.',
#     'Tomato___Leaf_Mold': 'Improve ventilation, apply fungicides like copper hydroxide.',
#     'Tomato___Septoria_leaf_spot': 'Use fungicides like azoxystrobin, remove lower leaves.',
#     'Tomato___Spider_mites Two-spotted_spider_mite': 'Apply miticides like abamectin, increase humidity.',
#     'Tomato___Target_Spot': 'Use fungicides like chlorothalonil, remove affected leaves.',
#     'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whiteflies with insecticides, remove infected plants.',
#     'Tomato___Tomato_mosaic_virus': 'Disinfect tools, remove and destroy infected plants.',
#     'Tomato___healthy': 'No treatment needed. Maintain proper spacing and nutrition.'
# }

# # Symptoms dictionary
# symptoms = {
#     'Apple___Apple_scab': 'Dark olive-green spots on leaves with velvety texture, later turning brown and cracked.',
#     'Apple___Black_rot': 'Small brown spots on leaves and fruit, expanding into larger rotted areas with black pycnidia.',
#     'Apple___Cedar_apple_rust': 'Yellow-orange spots on leaves with small tube-like structures, leading to leaf drop.',
#     'Apple___healthy': 'No visible symptoms. Leaves are green and uniformly healthy.',
#     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Grayish-white spots with brown margins on leaves, often merging.',
#     'Corn_(maize)___Common_rust_': 'Pustules of reddish-brown rust spores on both leaf surfaces, leading to drying.',
#     'Corn_(maize)___Northern_Leaf_Blight': 'Long, cigar-shaped gray-green lesions on leaves, turning tan as disease progresses.',
#     'Corn_(maize)___healthy': 'No symptoms. Leaves are green and robust with no spots or lesions.',
#     'Grape___Black_rot': 'Small brown spots on leaves and berries, expanding into larger rotted areas with black fungal growth.',
#     'Grape___Esca_(Black_Measles)': 'Tiger-striped discoloration on leaves, black margins, and fruit rot.',
#     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Dark brown to black spots with yellow halos on leaves.',
#     'Grape___healthy': 'No symptoms. Leaves are green and vines are vigorous.',
#     'Potato___Early_blight': 'Dark spots on leaves with circles around them and yellow rings. Leaves dry out and fall off.',
#     'Potato___Late_blight': 'Water-soaked spots on leaves turning brown with white fungal growth on undersides.',
#     'Potato___healthy': 'No symptoms. Plants show uniform green foliage and healthy tubers.',
#     'Tomato___Bacterial_spot': 'Small, dark brown to black spots on leaves and fruit, often with a yellow halo.',
#     'Tomato___Early_blight': 'Dark spots on leaves with circles around them and yellow rings. Leaves dry out and fall off.',
#     'Tomato___Late_blight': 'Water-soaked spots on leaves and stems turning brown, with white mold on undersides.',
#     'Tomato___Leaf_Mold': 'Yellowing on upper leaf surfaces with olive-green to brown mold on undersides.',
#     'Tomato___Septoria_leaf_spot': 'Small gray spots with dark borders on lower leaves, spreading upward.',
#     'Tomato___Spider_mites Two-spotted_spider_mite': 'Tiny yellow or white spots on leaves, webbing, and leaf stippling.',
#     'Tomato___Target_Spot': 'Dark brown spots with concentric rings on leaves, resembling a target.',
#     'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Upward leaf curling, yellowing, and stunted growth.',
#     'Tomato___Tomato_mosaic_virus': 'Mottled yellow and green patterns on leaves, stunted growth.',
#     'Tomato___healthy': 'No symptoms. Plants show uniform green leaves and healthy fruit.'
# }

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/community', methods=['GET', 'POST'])
# def community():
#     if request.method == 'POST' and current_user.is_authenticated:
#         title = request.form['title']
#         content = request.form['content']
#         image = request.files.get('image')
#         image_url = None
#         if image and allowed_file(image.filename):
#             filename = secure_filename(image.filename)
#             image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             image.save(image_path)
#             image_url = url_for('static', filename='uploads/' + filename)
#         post = Post(title=title, content=content, image_url=image_url, user_id=current_user.id)
#         db.session.add(post)
#         db.session.commit()
#         flash('Post created successfully!', 'success')
#         return redirect(url_for('community'))
    
#     posts = Post.query.order_by(Post.date.desc()).all()
#     for post in posts:
#         post.voted_by_current_user = False
#         if current_user.is_authenticated:
#             post.voted_by_current_user = PostVote.query.filter_by(
#                 user_id=current_user.id, post_id=post.id).first() is not None
#     return render_template('community.html', posts=posts)

# @app.route('/post/<int:post_id>', methods=['GET', 'POST'])
# def post_detail(post_id):
#     post = Post.query.get_or_404(post_id)
#     if request.method == 'POST' and current_user.is_authenticated:
#         content = request.form['content']
#         comment = Comment(content=content, post_id=post.id, user_id=current_user.id)
#         db.session.add(comment)
#         db.session.commit()
#         flash('Comment added successfully!', 'success')
#         return redirect(url_for('post_detail', post_id=post.id))
#     post.voted_by_current_user = False
#     if current_user.is_authenticated:
#         post.voted_by_current_user = PostVote.query.filter_by(
#             user_id=current_user.id, post_id=post.id).first() is not None
#     return render_template('post_detail.html', post=post)

# @app.route('/post/<int:post_id>/upvote')
# @login_required
# def upvote_post(post_id):
#     post = Post.query.get_or_404(post_id)
#     existing_vote = PostVote.query.filter_by(user_id=current_user.id, post_id=post_id).first()
#     if not existing_vote:
#         vote = PostVote(user_id=current_user.id, post_id=post_id, vote_type=1)
#         post.votes += 1
#         db.session.add(vote)
#         db.session.commit()
#         flash('Post upvoted!', 'success')
#     else:
#         flash('You have already voted on this post!', 'warning')
#     return redirect(request.referrer or url_for('community'))

# @app.route('/post/<int:post_id>/downvote')
# @login_required
# def downvote_post(post_id):
#     post = Post.query.get_or_404(post_id)
#     existing_vote = PostVote.query.filter_by(user_id=current_user.id, post_id=post_id).first()
#     if not existing_vote:
#         vote = PostVote(user_id=current_user.id, post_id=post_id, vote_type=-1)
#         post.votes -= 1
#         db.session.add(vote)
#         db.session.commit()
#         flash('Post downvoted!', 'success')
#     else:
#         flash('You have already voted on this post!', 'warning')
#     return redirect(request.referrer or url_for('community'))

# @app.route('/post/<int:post_id>/delete')
# @login_required
# def delete_post(post_id):
#     post = Post.query.get_or_404(post_id)
#     if post.user_id != current_user.id:
#         flash('You can only delete your own posts!', 'error')
#         return redirect(url_for('community'))
    
#     PostVote.query.filter_by(post_id=post_id).delete()
#     Comment.query.filter_by(post_id=post_id).delete()
    
#     if post.image_url:
#         image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(post.image_url))
#         if os.path.exists(image_path):
#             os.remove(image_path)
#     db.session.delete(post)
#     db.session.commit()
#     flash('Post deleted successfully!', 'success')
#     return redirect(url_for('community'))

# @app.route('/comment/<int:comment_id>/upvote')
# @login_required
# def upvote_comment(comment_id):
#     comment = Comment.query.get_or_404(comment_id)
#     existing_vote = CommentVote.query.filter_by(user_id=current_user.id, comment_id=comment_id).first()
#     if not existing_vote:
#         vote = CommentVote(user_id=current_user.id, comment_id=comment_id, vote_type=1)
#         comment.votes += 1
#         db.session.add(vote)
#         db.session.commit()
#         flash('Comment upvoted!', 'success')
#     else:
#         flash('You have already voted on this comment!', 'warning')
#     return redirect(request.referrer or url_for('post_detail', post_id=comment.post_id))

# @app.route('/comment/<int:comment_id>/downvote')
# @login_required
# def downvote_comment(comment_id):
#     comment = Comment.query.get_or_404(comment_id)
#     existing_vote = CommentVote.query.filter_by(user_id=current_user.id, comment_id=comment_id).first()
#     if not existing_vote:
#         vote = CommentVote(user_id=current_user.id, comment_id=comment_id, vote_type=-1)
#         comment.votes -= 1
#         db.session.add(vote)
#         db.session.commit()
#         flash('Comment downvoted!', 'success')
#     else:
#         flash('You have already voted on this comment!', 'warning')
#     return redirect(request.referrer or url_for('post_detail', post_id=comment.post_id))

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         user = User.query.filter_by(username=username).first()
#         if user and user.check_password(password):
#             login_user(user)
#             flash('Logged in successfully!', 'success')
#             return redirect(url_for('predict'))
#         else:
#             flash('Invalid username or password.', 'error')
#     return render_template('login.html')

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         if User.query.filter_by(username=username).first():
#             flash('Username already exists.', 'error')
#         else:
#             user = User(username=username)
#             user.set_password(password)
#             db.session.add(user)
#             db.session.commit()
#             flash('Registered successfully! Please log in.', 'success')
#             return redirect(url_for('login'))
#     return render_template('register.html')

# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     flash('Logged out successfully.', 'success')
#     return redirect(url_for('home'))

# @app.route('/predict', methods=['GET', 'POST'])
# @login_required
# @app.route('/predict', methods=['GET', 'POST'])
# @login_required
# def predict():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             flash('No file part', 'error')
#             return redirect(request.url)
        
#         file = request.files['file']
        
#         if file.filename == '':
#             flash('No selected file', 'error')
#             return redirect(request.url)
        
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)

#             try:
#                 result_index, input_arr = model_prediction(file_path)
#                 raw_class = class_name[result_index]
#                 predicted_class = format_class_name(raw_class)
#                 treatment = treatments.get(raw_class, "No treatment information available.")
#                 symptom = symptoms.get(raw_class, "No symptom information available.")
#                 image_url = url_for('static', filename='uploads/' + filename)

#                 # Generate SHAP explanation
#                 shap_image = get_shap_explanation(input_arr, result_index)
#                 print(f"SHAP image result: {shap_image[:50] if shap_image else 'None'}...")  # Truncate for readability

#                 if 'share_to_community' in request.form:
#                     title = f"Prediction: {predicted_class}"
#                     content = f"I got this prediction: {predicted_class}\nSymptoms: {symptom}\nTreatment: {treatment}"
#                     post = Post(title=title, content=content, image_url=image_url, user_id=current_user.id)
#                     db.session.add(post)
#                     db.session.commit()
#                     flash('Prediction shared to community!', 'success')
#                     return redirect(url_for('community'))

#                 return render_template('predict.html', image_url=image_url, prediction=predicted_class, 
#                                      treatment=treatment, symptom=symptom, shap_image=shap_image)
#             except Exception as e:
#                 flash(f'Prediction error: {str(e)}', 'error')
#                 return redirect(request.url)

#     return render_template('predict.html')
# if __name__ == '__main__':
#     app.run(debug=True)
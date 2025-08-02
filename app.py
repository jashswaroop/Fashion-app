from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import base64
import io
from datetime import datetime
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fashion_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Profile fields
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    body_type = db.Column(db.String(50))
    style_preference = db.Column(db.String(100))
    color_preference = db.Column(db.String(200))
    size_preference = db.Column(db.String(50))
    budget_range = db.Column(db.String(50))
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# User Image Model
class UserImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    image_type = db.Column(db.String(50), nullable=False)  # 'face', 'full_body', 'outfit'
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)

    # Analysis results
    face_shape = db.Column(db.String(50))
    skin_tone = db.Column(db.String(50))
    body_type_detected = db.Column(db.String(50))
    dominant_colors = db.Column(db.Text)  # JSON string of colors
    analysis_complete = db.Column(db.Boolean, default=False)

    user = db.relationship('User', backref=db.backref('images', lazy=True))

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_face_shape(image_path):
    """Detect face shape using improved OpenCV analysis"""
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return "Unknown"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load face cascade with better parameters
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces with multiple scale factors for better accuracy
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            # Try with more relaxed parameters
            faces = face_cascade.detectMultiScale(gray, 1.3, 3, minSize=(50, 50))
            if len(faces) == 0:
                return "Unknown"

        # Get the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face

        # Extract face region with some padding
        padding = int(min(w, h) * 0.1)
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)

        face_region = gray[y_start:y_end, x_start:x_end]
        face_color = img[y_start:y_end, x_start:x_end]

        # Analyze face shape using multiple methods
        face_shape = analyze_face_geometry(face_region, w, h)
        return face_shape

    except Exception as e:
        print(f"Error in face detection: {e}")
        return "Unknown"

def analyze_face_geometry(face_region, face_width, face_height):
    """Analyze face geometry to determine shape"""
    try:
        # Calculate basic ratios
        width_height_ratio = face_width / face_height

        # Apply edge detection to find face contours
        blurred = cv2.GaussianBlur(face_region, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return classify_by_ratio(width_height_ratio)

        # Get the largest contour (likely face outline)
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate contour properties
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        # Calculate circularity (4π * area / perimeter²)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0

        # Calculate bounding rectangle
        rect = cv2.boundingRect(largest_contour)
        rect_width, rect_height = rect[2], rect[3]
        rect_ratio = rect_width / rect_height if rect_height > 0 else 1

        # Analyze face regions (upper, middle, lower thirds)
        upper_third = face_region[:face_region.shape[0]//3, :]
        middle_third = face_region[face_region.shape[0]//3:2*face_region.shape[0]//3, :]
        lower_third = face_region[2*face_region.shape[0]//3:, :]

        # Calculate width variations across face thirds
        upper_width = analyze_region_width(upper_third)
        middle_width = analyze_region_width(middle_third)
        lower_width = analyze_region_width(lower_third)

        # Determine face shape based on multiple factors
        return classify_face_shape_advanced(
            width_height_ratio, circularity, rect_ratio,
            upper_width, middle_width, lower_width
        )

    except Exception as e:
        print(f"Error in geometry analysis: {e}")
        return classify_by_ratio(face_width / face_height)

def analyze_region_width(region):
    """Analyze the effective width of a face region"""
    try:
        if region.size == 0:
            return 0

        # Apply edge detection
        edges = cv2.Canny(region, 50, 150)

        # Find the widest part by analyzing each row
        max_width = 0
        for row in edges:
            # Find leftmost and rightmost edge pixels
            edge_pixels = np.where(row > 0)[0]
            if len(edge_pixels) > 0:
                width = edge_pixels[-1] - edge_pixels[0]
                max_width = max(max_width, width)

        return max_width
    except:
        return 0

def classify_face_shape_advanced(width_height_ratio, circularity, rect_ratio,
                                upper_width, middle_width, lower_width):
    """Advanced face shape classification using multiple geometric factors"""

    # Normalize widths to avoid division by zero
    total_width = max(upper_width + middle_width + lower_width, 1)
    upper_ratio = upper_width / total_width
    middle_ratio = middle_width / total_width
    lower_ratio = lower_width / total_width

    # Classification logic based on multiple factors

    # ROUND FACE: High circularity, similar width/height, even width distribution
    if (circularity > 0.6 and
        0.85 <= width_height_ratio <= 1.15 and
        abs(upper_ratio - middle_ratio) < 0.1 and
        abs(middle_ratio - lower_ratio) < 0.1):
        return "Round"

    # OVAL FACE: Lower width/height ratio, balanced proportions
    elif (0.7 <= width_height_ratio <= 0.9 and
          middle_ratio > upper_ratio and middle_ratio > lower_ratio and
          abs(upper_ratio - lower_ratio) < 0.15):
        return "Oval"

    # SQUARE FACE: Similar width/height, angular (low circularity), even width
    elif (0.85 <= width_height_ratio <= 1.1 and
          circularity < 0.6 and
          abs(upper_ratio - middle_ratio) < 0.1 and
          abs(middle_ratio - lower_ratio) < 0.1):
        return "Square"

    # HEART FACE: Wider forehead, narrower jaw
    elif (upper_ratio > middle_ratio and
          middle_ratio > lower_ratio and
          upper_ratio - lower_ratio > 0.15):
        return "Heart"

    # DIAMOND FACE: Wider cheekbones, narrower forehead and jaw
    elif (middle_ratio > upper_ratio and
          middle_ratio > lower_ratio and
          abs(upper_ratio - lower_ratio) < 0.1):
        return "Diamond"

    # Fallback to ratio-based classification
    else:
        return classify_by_ratio(width_height_ratio)

def classify_by_ratio(ratio):
    """Simple fallback classification based on width/height ratio"""
    if ratio >= 1.0:
        return "Round"
    elif ratio <= 0.75:
        return "Oval"
    elif 0.85 <= ratio < 1.0:
        return "Square"
    else:
        return "Heart"

def detect_face_shape_basic(image_path):
    """Basic face shape detection using OpenCV (fallback method)"""
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return "Unknown"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))

        if len(faces) == 0:
            return "Unknown"

        # Get the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face

        # Extract face region for more detailed analysis
        face_region = gray[y:y+h, x:x+w]

        # Calculate multiple ratios for better accuracy
        face_ratio = w / h

        # Analyze face contours for better shape detection
        # Apply Gaussian blur and find contours
        blurred = cv2.GaussianBlur(face_region, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour (likely the face outline)
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate contour properties
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)

            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
        else:
            circularity = 0

        # Improved classification with multiple factors
        # Round: High circularity, ratio close to 1
        if circularity > 0.7 and 0.9 <= face_ratio <= 1.1:
            return "Round"
        # Oval: Lower circularity, ratio between 0.7-0.9
        elif 0.4 <= circularity <= 0.7 and 0.7 <= face_ratio <= 0.9:
            return "Oval"
        # Square: Medium circularity, ratio close to 1 but more angular
        elif 0.5 <= circularity <= 0.8 and 0.85 <= face_ratio <= 1.0:
            return "Square"
        # Heart: Lower circularity, ratio varies
        elif circularity < 0.6 and 0.75 <= face_ratio <= 0.95:
            return "Heart"
        # Default classification based on ratio
        elif face_ratio >= 1.0:
            return "Round"
        elif face_ratio <= 0.75:
            return "Oval"
        elif 0.85 <= face_ratio < 1.0:
            return "Square"
        else:
            return "Heart"

    except Exception as e:
        print(f"Error in basic face detection: {e}")
        return "Unknown"



def analyze_skin_tone(image_path):
    """Analyze skin tone from image"""
    try:
        img = cv2.imread(image_path)

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return "Unknown"

        # Get face region
        x, y, w, h = faces[0]
        face_region = img_rgb[y:y+h, x:x+w]

        # Calculate average color in face region
        avg_color = np.mean(face_region.reshape(-1, 3), axis=0)

        # Simple skin tone classification
        r, g, b = avg_color

        if r > 200 and g > 180 and b > 170:
            return "Fair"
        elif r > 160 and g > 120 and b > 100:
            return "Medium"
        elif r > 120 and g > 80 and b > 60:
            return "Olive"
        else:
            return "Deep"

    except Exception as e:
        print(f"Error in skin tone analysis: {e}")
        return "Unknown"

def extract_dominant_colors(image_path, k=5):
    """Extract dominant colors from image"""
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Reshape image to be a list of pixels
        pixels = img.reshape(-1, 3)

        # Use K-means clustering to find dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pixels)

        # Get the colors
        colors = kmeans.cluster_centers_.astype(int)

        # Convert to hex colors
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in colors]

        return hex_colors

    except Exception as e:
        print(f"Error in color extraction: {e}")
        return []

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Template filter for JSON parsing
@app.template_filter('from_json')
def from_json_filter(value):
    try:
        return json.loads(value) if value else []
    except:
        return []

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return render_template('register.html')
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.age = request.form.get('age')
        current_user.gender = request.form.get('gender')
        current_user.body_type = request.form.get('body_type')
        current_user.style_preference = request.form.get('style_preference')
        current_user.color_preference = request.form.get('color_preference')
        current_user.size_preference = request.form.get('size_preference')
        current_user.budget_range = request.form.get('budget_range')
        
        db.session.commit()
        flash('Profile updated successfully!')
        return redirect(url_for('profile'))
    
    return render_template('profile.html', user=current_user)

@app.route('/image_capture')
@login_required
def image_capture():
    return render_template('image_capture.html')

@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    image_type = request.form.get('image_type', 'face')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        # Generate secure filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename

        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Create database record
        user_image = UserImage(
            user_id=current_user.id,
            filename=filename,
            original_filename=file.filename,
            image_type=image_type
        )
        db.session.add(user_image)
        db.session.commit()

        # Start analysis in background (simplified for demo)
        try:
            if image_type == 'face':
                face_shape = detect_face_shape(filepath)
                skin_tone = analyze_skin_tone(filepath)
                user_image.face_shape = face_shape
                user_image.skin_tone = skin_tone

            # Extract dominant colors for all image types
            colors = extract_dominant_colors(filepath)
            user_image.dominant_colors = json.dumps(colors)
            user_image.analysis_complete = True

            db.session.commit()

            return jsonify({
                'success': True,
                'image_id': user_image.id,
                'filename': filename,
                'analysis': {
                    'face_shape': user_image.face_shape,
                    'skin_tone': user_image.skin_tone,
                    'dominant_colors': colors
                }
            })

        except Exception as e:
            print(f"Analysis error: {e}")
            return jsonify({
                'success': True,
                'image_id': user_image.id,
                'filename': filename,
                'analysis': None,
                'message': 'Image uploaded but analysis failed'
            })

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/webcam_capture', methods=['POST'])
@login_required
def webcam_capture():
    try:
        # Get the image data from the request
        image_data = request.json.get('image_data')
        image_type = request.json.get('image_type', 'face')

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Remove the data URL prefix
        image_data = image_data.split(',')[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'webcam_{timestamp}.png'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save image
        image.save(filepath)

        # Create database record
        user_image = UserImage(
            user_id=current_user.id,
            filename=filename,
            original_filename=f'webcam_capture_{timestamp}.png',
            image_type=image_type
        )
        db.session.add(user_image)
        db.session.commit()

        # Perform analysis
        try:
            if image_type == 'face':
                face_shape = detect_face_shape(filepath)
                skin_tone = analyze_skin_tone(filepath)
                user_image.face_shape = face_shape
                user_image.skin_tone = skin_tone

            colors = extract_dominant_colors(filepath)
            user_image.dominant_colors = json.dumps(colors)
            user_image.analysis_complete = True

            db.session.commit()

            return jsonify({
                'success': True,
                'image_id': user_image.id,
                'filename': filename,
                'analysis': {
                    'face_shape': user_image.face_shape,
                    'skin_tone': user_image.skin_tone,
                    'dominant_colors': colors
                }
            })

        except Exception as e:
            print(f"Analysis error: {e}")
            return jsonify({
                'success': True,
                'image_id': user_image.id,
                'filename': filename,
                'analysis': None,
                'message': 'Image captured but analysis failed'
            })

    except Exception as e:
        print(f"Webcam capture error: {e}")
        return jsonify({'error': 'Failed to process webcam image'}), 500

@app.route('/my_images')
@login_required
def my_images():
    images = UserImage.query.filter_by(user_id=current_user.id).order_by(UserImage.upload_date.desc()).all()
    return render_template('my_images.html', images=images)

@app.route('/delete_image/<int:image_id>', methods=['POST'])
@login_required
def delete_image(image_id):
    image = UserImage.query.filter_by(id=image_id, user_id=current_user.id).first()
    if image:
        # Delete file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        if os.path.exists(filepath):
            os.remove(filepath)

        # Delete database record
        db.session.delete(image)
        db.session.commit()

        flash('Image deleted successfully!')
    else:
        flash('Image not found!')

    return redirect(url_for('my_images'))

@app.route('/recommendations')
@login_required
def recommendations():
    # Get user's images with analysis
    face_images = UserImage.query.filter_by(
        user_id=current_user.id,
        image_type='face',
        analysis_complete=True
    ).all()

    # Get the most recent face analysis
    latest_face_analysis = None
    if face_images:
        latest_face_analysis = face_images[0]

    # Generate recommendations based on face shape and user profile
    recommendations_data = generate_face_shape_recommendations(
        latest_face_analysis, current_user
    )

    # Generate color recommendations based on skin tone
    color_recommendations = generate_color_recommendations(latest_face_analysis)

    return render_template('recommendations.html',
                         recommendations=recommendations_data,
                         color_recommendations=color_recommendations,
                         face_analysis=latest_face_analysis,
                         user=current_user)

@app.route('/color_analysis')
@login_required
def color_analysis():
    # Get user's images with analysis
    face_images = UserImage.query.filter_by(
        user_id=current_user.id,
        image_type='face',
        analysis_complete=True
    ).all()

    latest_face_analysis = None
    if face_images:
        latest_face_analysis = face_images[0]

    color_recommendations = generate_color_recommendations(latest_face_analysis)

    return render_template('color_analysis.html',
                         color_recommendations=color_recommendations,
                         face_analysis=latest_face_analysis,
                         user=current_user)

def generate_face_shape_recommendations(face_analysis, user):
    """Generate personalized recommendations based on face shape and profile"""
    recommendations = {
        'hairstyles': [],
        'glasses': [],
        'accessories': [],
        'makeup_tips': [],
        'clothing_necklines': []
    }

    if not face_analysis or not face_analysis.face_shape:
        return recommendations

    face_shape = face_analysis.face_shape.lower()

    # Hairstyle recommendations based on face shape
    hairstyle_recommendations = {
        'round': [
            {'name': 'Long Layers', 'description': 'Add length and angles to elongate your face', 'image': 'long-layers.jpg'},
            {'name': 'Side-Swept Bangs', 'description': 'Create asymmetry and add height', 'image': 'side-bangs.jpg'},
            {'name': 'High Ponytail', 'description': 'Adds height and elongates the face', 'image': 'high-ponytail.jpg'},
            {'name': 'Angular Bob', 'description': 'Sharp angles complement round features', 'image': 'angular-bob.jpg'}
        ],
        'oval': [
            {'name': 'Almost Any Style', 'description': 'Your balanced proportions work with most cuts', 'image': 'versatile.jpg'},
            {'name': 'Blunt Bob', 'description': 'Classic and timeless for oval faces', 'image': 'blunt-bob.jpg'},
            {'name': 'Curtain Bangs', 'description': 'Frame your face beautifully', 'image': 'curtain-bangs.jpg'},
            {'name': 'Pixie Cut', 'description': 'Bold and chic option', 'image': 'pixie-cut.jpg'}
        ],
        'square': [
            {'name': 'Soft Waves', 'description': 'Soften angular jawline with gentle curves', 'image': 'soft-waves.jpg'},
            {'name': 'Long Layers', 'description': 'Add movement and soften edges', 'image': 'long-layers-square.jpg'},
            {'name': 'Side Part', 'description': 'Asymmetry balances strong jaw', 'image': 'side-part.jpg'},
            {'name': 'Rounded Bob', 'description': 'Curved lines complement angular features', 'image': 'rounded-bob.jpg'}
        ],
        'heart': [
            {'name': 'Chin-Length Bob', 'description': 'Adds width at the jawline', 'image': 'chin-bob.jpg'},
            {'name': 'Full Bangs', 'description': 'Balance a wider forehead', 'image': 'full-bangs.jpg'},
            {'name': 'Textured Lob', 'description': 'Creates volume at the bottom', 'image': 'textured-lob.jpg'},
            {'name': 'Side-Swept Layers', 'description': 'Soften the forehead area', 'image': 'side-layers.jpg'}
        ],
        'diamond': [
            {'name': 'Side-Parted Waves', 'description': 'Softens angular cheekbones', 'image': 'side-waves.jpg'},
            {'name': 'Chin-Length Layers', 'description': 'Adds width to narrow jaw and forehead', 'image': 'chin-layers.jpg'},
            {'name': 'Soft Bangs', 'description': 'Widens the forehead area', 'image': 'soft-bangs.jpg'},
            {'name': 'Textured Pixie', 'description': 'Emphasizes cheekbones beautifully', 'image': 'textured-pixie.jpg'}
        ]
    }

    # Glasses recommendations
    glasses_recommendations = {
        'round': [
            {'name': 'Angular Frames', 'description': 'Square or rectangular frames add definition', 'style': 'rectangular'},
            {'name': 'Cat-Eye Glasses', 'description': 'Upswept frames elongate the face', 'style': 'cat-eye'},
            {'name': 'Geometric Shapes', 'description': 'Sharp angles contrast round features', 'style': 'geometric'}
        ],
        'oval': [
            {'name': 'Aviator Style', 'description': 'Classic style that complements balanced features', 'style': 'aviator'},
            {'name': 'Round Frames', 'description': 'Soft curves work well with oval faces', 'style': 'round'},
            {'name': 'Square Frames', 'description': 'Add structure to soft features', 'style': 'square'}
        ],
        'square': [
            {'name': 'Round Frames', 'description': 'Soften angular features with curved lines', 'style': 'round'},
            {'name': 'Oval Frames', 'description': 'Gentle curves balance strong jaw', 'style': 'oval'},
            {'name': 'Cat-Eye Style', 'description': 'Upswept frames add femininity', 'style': 'cat-eye'}
        ],
        'heart': [
            {'name': 'Bottom-Heavy Frames', 'description': 'Add width to the lower face', 'style': 'bottom-heavy'},
            {'name': 'Round Frames', 'description': 'Soften the forehead area', 'style': 'round'},
            {'name': 'Rimless Glasses', 'description': 'Minimize emphasis on forehead', 'style': 'rimless'}
        ],
        'diamond': [
            {'name': 'Oval Frames', 'description': 'Soften angular cheekbones', 'style': 'oval'},
            {'name': 'Cat-Eye Glasses', 'description': 'Add width to forehead and jaw', 'style': 'cat-eye'},
            {'name': 'Browline Frames', 'description': 'Emphasize the upper face', 'style': 'browline'}
        ]
    }

    # Accessory recommendations
    accessory_recommendations = {
        'round': [
            {'name': 'Long Earrings', 'description': 'Elongate the face with vertical lines'},
            {'name': 'Angular Necklaces', 'description': 'Geometric shapes add definition'},
            {'name': 'Structured Scarves', 'description': 'Sharp folds create angles'}
        ],
        'oval': [
            {'name': 'Statement Earrings', 'description': 'Bold pieces complement balanced features'},
            {'name': 'Choker Necklaces', 'description': 'Highlight the neckline beautifully'},
            {'name': 'Wide Headbands', 'description': 'Frame the face elegantly'}
        ],
        'square': [
            {'name': 'Hoop Earrings', 'description': 'Soft curves balance angular jaw'},
            {'name': 'Layered Necklaces', 'description': 'Create visual interest and softness'},
            {'name': 'Flowing Scarves', 'description': 'Soft draping complements strong features'}
        ],
        'heart': [
            {'name': 'Chandelier Earrings', 'description': 'Add width at the jawline'},
            {'name': 'Statement Necklaces', 'description': 'Draw attention to the lower face'},
            {'name': 'Wide Brims', 'description': 'Balance a wider forehead'}
        ],
        'diamond': [
            {'name': 'Stud Earrings', 'description': 'Don\'t compete with strong cheekbones'},
            {'name': 'Delicate Necklaces', 'description': 'Keep focus on facial features'},
            {'name': 'Soft Scarves', 'description': 'Add width to forehead and jaw areas'}
        ]
    }

    # Clothing neckline recommendations
    neckline_recommendations = {
        'round': [
            {'name': 'V-Neck', 'description': 'Creates vertical lines to elongate'},
            {'name': 'Scoop Neck', 'description': 'Opens up the chest area'},
            {'name': 'Off-Shoulder', 'description': 'Draws attention horizontally'}
        ],
        'oval': [
            {'name': 'Boat Neck', 'description': 'Highlights balanced proportions'},
            {'name': 'Turtleneck', 'description': 'Frames the face beautifully'},
            {'name': 'Strapless', 'description': 'Shows off elegant neckline'}
        ],
        'square': [
            {'name': 'Rounded Necklines', 'description': 'Soften angular features'},
            {'name': 'Cowl Neck', 'description': 'Draping adds softness'},
            {'name': 'Sweetheart', 'description': 'Curved lines complement jaw'}
        ],
        'heart': [
            {'name': 'Boat Neck', 'description': 'Balances wider forehead'},
            {'name': 'Square Neck', 'description': 'Adds width to lower face'},
            {'name': 'Halter Top', 'description': 'Creates horizontal emphasis'}
        ],
        'diamond': [
            {'name': 'Scoop Neck', 'description': 'Softens angular features'},
            {'name': 'V-Neck', 'description': 'Draws attention away from cheekbones'},
            {'name': 'Wrap Style', 'description': 'Creates diagonal lines that flatter'}
        ]
    }

    # Get recommendations for the detected face shape
    recommendations['hairstyles'] = hairstyle_recommendations.get(face_shape, [])
    recommendations['glasses'] = glasses_recommendations.get(face_shape, [])
    recommendations['accessories'] = accessory_recommendations.get(face_shape, [])
    recommendations['clothing_necklines'] = neckline_recommendations.get(face_shape, [])

    # Add makeup tips based on face shape
    makeup_tips = {
        'round': [
            'Contour the sides of your face to add definition',
            'Use highlighter on the center of your face',
            'Apply blush on the apples of your cheeks',
            'Use darker eyeshadow on outer corners to elongate eyes'
        ],
        'oval': [
            'You can experiment with most makeup styles',
            'Highlight your natural bone structure',
            'Try bold lip colors to make a statement',
            'Experiment with different eyeshadow techniques'
        ],
        'square': [
            'Soften your jawline with contouring',
            'Use rounded blush application',
            'Highlight the center of your forehead and chin',
            'Try soft, smoky eye looks'
        ],
        'heart': [
            'Contour your forehead to minimize width',
            'Highlight your chin to add width',
            'Use blush on the apples of your cheeks',
            'Try bold lip colors to balance your features'
        ],
        'diamond': [
            'Highlight your forehead and chin to add width',
            'Contour your cheekbones to soften them',
            'Use blush below the cheekbones',
            'Try neutral lip colors to keep focus balanced'
        ]
    }

    recommendations['makeup_tips'] = makeup_tips.get(face_shape, [])

    return recommendations

def generate_color_recommendations(face_analysis):
    """Generate color recommendations based on skin tone"""
    color_recommendations = {
        'best_colors': [],
        'avoid_colors': [],
        'neutral_colors': [],
        'accent_colors': [],
        'seasonal_palette': '',
        'makeup_colors': {}
    }

    if not face_analysis or not face_analysis.skin_tone:
        return color_recommendations

    skin_tone = face_analysis.skin_tone.lower()

    # Color recommendations based on skin tone
    color_palettes = {
        'fair': {
            'best_colors': [
                {'name': 'Soft Pink', 'hex': '#F8BBD9', 'description': 'Complements fair skin beautifully'},
                {'name': 'Lavender', 'hex': '#E6E6FA', 'description': 'Enhances cool undertones'},
                {'name': 'Mint Green', 'hex': '#98FB98', 'description': 'Fresh and flattering'},
                {'name': 'Powder Blue', 'hex': '#B0E0E6', 'description': 'Soft and elegant'},
                {'name': 'Cream', 'hex': '#F5F5DC', 'description': 'Classic and versatile'}
            ],
            'avoid_colors': [
                {'name': 'Bright Orange', 'hex': '#FF4500', 'reason': 'Can wash out fair skin'},
                {'name': 'Neon Yellow', 'hex': '#FFFF00', 'reason': 'Too harsh for delicate coloring'},
                {'name': 'Hot Pink', 'hex': '#FF1493', 'reason': 'Can clash with cool undertones'}
            ],
            'seasonal_palette': 'Summer',
            'makeup_colors': {
                'lipstick': ['Rose Pink', 'Berry', 'Coral Pink'],
                'eyeshadow': ['Soft Brown', 'Taupe', 'Champagne'],
                'blush': ['Peach', 'Rose', 'Pink']
            }
        },
        'medium': {
            'best_colors': [
                {'name': 'Emerald Green', 'hex': '#50C878', 'description': 'Rich and vibrant'},
                {'name': 'Royal Blue', 'hex': '#4169E1', 'description': 'Classic and striking'},
                {'name': 'Deep Purple', 'hex': '#663399', 'description': 'Elegant and sophisticated'},
                {'name': 'Burgundy', 'hex': '#800020', 'description': 'Rich and warm'},
                {'name': 'Golden Yellow', 'hex': '#FFD700', 'description': 'Warm and radiant'}
            ],
            'avoid_colors': [
                {'name': 'Pale Pink', 'hex': '#FFB6C1', 'reason': 'Can look washed out'},
                {'name': 'Light Gray', 'hex': '#D3D3D3', 'reason': 'May appear dull'},
                {'name': 'Beige', 'hex': '#F5F5DC', 'reason': 'Can blend too much with skin'}
            ],
            'seasonal_palette': 'Autumn',
            'makeup_colors': {
                'lipstick': ['Red', 'Plum', 'Bronze'],
                'eyeshadow': ['Gold', 'Bronze', 'Deep Brown'],
                'blush': ['Coral', 'Terracotta', 'Warm Pink']
            }
        },
        'olive': {
            'best_colors': [
                {'name': 'Forest Green', 'hex': '#228B22', 'description': 'Harmonizes with olive undertones'},
                {'name': 'Burnt Orange', 'hex': '#CC5500', 'description': 'Warm and complementary'},
                {'name': 'Deep Teal', 'hex': '#008080', 'description': 'Rich and sophisticated'},
                {'name': 'Chocolate Brown', 'hex': '#7B3F00', 'description': 'Earthy and natural'},
                {'name': 'Mustard Yellow', 'hex': '#FFDB58', 'description': 'Warm and vibrant'}
            ],
            'avoid_colors': [
                {'name': 'Bright Pink', 'hex': '#FF1493', 'reason': 'Can clash with green undertones'},
                {'name': 'Cool Blue', 'hex': '#0000FF', 'reason': 'May look harsh'},
                {'name': 'Pure White', 'hex': '#FFFFFF', 'reason': 'Can be too stark'}
            ],
            'seasonal_palette': 'Autumn',
            'makeup_colors': {
                'lipstick': ['Brick Red', 'Brown', 'Orange Red'],
                'eyeshadow': ['Olive', 'Bronze', 'Copper'],
                'blush': ['Peach', 'Coral', 'Warm Brown']
            }
        },
        'deep': {
            'best_colors': [
                {'name': 'Bright White', 'hex': '#FFFFFF', 'description': 'Creates beautiful contrast'},
                {'name': 'Electric Blue', 'hex': '#7DF9FF', 'description': 'Vibrant and striking'},
                {'name': 'Hot Pink', 'hex': '#FF1493', 'description': 'Bold and beautiful'},
                {'name': 'Emerald Green', 'hex': '#50C878', 'description': 'Rich and luxurious'},
                {'name': 'Golden Yellow', 'hex': '#FFD700', 'description': 'Warm and radiant'}
            ],
            'avoid_colors': [
                {'name': 'Muddy Brown', 'hex': '#8B4513', 'reason': 'Can look dull'},
                {'name': 'Olive Green', 'hex': '#808000', 'reason': 'May blend too much'},
                {'name': 'Dusty Pink', 'hex': '#DC143C', 'reason': 'Can appear washed out'}
            ],
            'seasonal_palette': 'Winter',
            'makeup_colors': {
                'lipstick': ['Deep Red', 'Fuchsia', 'Berry'],
                'eyeshadow': ['Deep Purple', 'Navy', 'Gold'],
                'blush': ['Deep Rose', 'Plum', 'Berry']
            }
        }
    }

    # Get recommendations for the detected skin tone
    if skin_tone in color_palettes:
        palette = color_palettes[skin_tone]
        color_recommendations['best_colors'] = palette['best_colors']
        color_recommendations['avoid_colors'] = palette['avoid_colors']
        color_recommendations['seasonal_palette'] = palette['seasonal_palette']
        color_recommendations['makeup_colors'] = palette['makeup_colors']

        # Add neutral colors that work for all skin tones
        color_recommendations['neutral_colors'] = [
            {'name': 'Navy Blue', 'hex': '#000080', 'description': 'Classic and versatile'},
            {'name': 'Charcoal Gray', 'hex': '#36454F', 'description': 'Sophisticated neutral'},
            {'name': 'Black', 'hex': '#000000', 'description': 'Timeless and elegant'}
        ]

        # Add accent colors based on seasonal palette
        accent_colors = {
            'Summer': [
                {'name': 'Soft Coral', 'hex': '#FF7F7F'},
                {'name': 'Periwinkle', 'hex': '#CCCCFF'}
            ],
            'Autumn': [
                {'name': 'Rust', 'hex': '#B7410E'},
                {'name': 'Olive', 'hex': '#808000'}
            ],
            'Winter': [
                {'name': 'Magenta', 'hex': '#FF00FF'},
                {'name': 'Turquoise', 'hex': '#40E0D0'}
            ]
        }

        color_recommendations['accent_colors'] = accent_colors.get(palette['seasonal_palette'], [])

    return color_recommendations

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

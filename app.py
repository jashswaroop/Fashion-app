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

    # Fashion preference analysis
    fashion_preferences = db.Column(db.Text)  # JSON string of detailed preferences
    style_analysis_complete = db.Column(db.Boolean, default=False)
    
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
    """Detect face shape using advanced facial analysis with multiple methods"""
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return "Unknown"

        # Convert to RGB for better processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try multiple face detection methods for better accuracy
        face_data = detect_face_with_multiple_methods(img, gray)

        if face_data is None:
            print("No face detected in image")
            return "Unknown"

        x, y, w, h = face_data

        # Extract face region with proper padding
        padding_x = int(w * 0.15)  # 15% padding
        padding_y = int(h * 0.15)

        x_start = max(0, x - padding_x)
        y_start = max(0, y - padding_y)
        x_end = min(img.shape[1], x + w + padding_x)
        y_end = min(img.shape[0], y + h + padding_y)

        face_region_gray = gray[y_start:y_end, x_start:x_end]
        face_region_color = img_rgb[y_start:y_end, x_start:x_end]

        # Perform comprehensive face shape analysis
        face_shape = analyze_face_shape_comprehensive(
            face_region_gray, face_region_color, w, h
        )

        print(f"Detected face shape: {face_shape}")
        return face_shape

    except Exception as e:
        print(f"Error in face shape detection: {e}")
        import traceback
        traceback.print_exc()
        return "Unknown"

def detect_face_with_multiple_methods(img, gray):
    """Try multiple face detection methods for better accuracy"""

    # Method 1: Haar Cascade (frontal face)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Try different parameters for better detection
    detection_params = [
        {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (80, 80)},
        {'scaleFactor': 1.2, 'minNeighbors': 4, 'minSize': (60, 60)},
        {'scaleFactor': 1.3, 'minNeighbors': 3, 'minSize': (40, 40)},
    ]

    for params in detection_params:
        faces = face_cascade.detectMultiScale(gray, **params)
        if len(faces) > 0:
            # Return the largest face
            return max(faces, key=lambda x: x[2] * x[3])

    # Method 2: Profile face detection as fallback
    try:
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        faces = profile_cascade.detectMultiScale(gray, 1.3, 3, minSize=(40, 40))
        if len(faces) > 0:
            return max(faces, key=lambda x: x[2] * x[3])
    except:
        pass

    return None

def analyze_face_shape_comprehensive(face_gray, face_color=None, face_width=0, face_height=0):
    """Comprehensive face shape analysis using multiple geometric features"""
    try:
        # Basic measurements
        width_height_ratio = face_width / face_height

        # 1. Analyze face contour using adaptive thresholding
        contour_features = analyze_face_contour(face_gray)

        # 2. Analyze facial proportions using golden ratio principles
        proportion_features = analyze_facial_proportions(face_gray)

        # 3. Analyze jawline characteristics
        jawline_features = analyze_jawline_shape(face_gray)

        # 4. Analyze forehead characteristics
        forehead_features = analyze_forehead_shape(face_gray)

        # 5. Combine all features for classification
        face_shape = classify_face_shape_ml_approach(
            width_height_ratio,
            contour_features,
            proportion_features,
            jawline_features,
            forehead_features
        )

        return face_shape

    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        # Fallback to basic ratio analysis
        return classify_by_ratio_improved(face_width / face_height)

def analyze_face_contour(face_gray):
    """Analyze face contour characteristics"""
    try:
        # Enhance contrast for better edge detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(face_gray)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Use adaptive threshold for better contour detection
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {'circularity': 0.5, 'aspect_ratio': 1.0, 'solidity': 0.5}

        # Get the largest contour (face outline)
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate contour properties
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        # Circularity: 4π * area / perimeter²
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # Aspect ratio of bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1

        # Solidity: contour area / convex hull area
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        return {
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity
        }

    except Exception as e:
        print(f"Error in contour analysis: {e}")
        return {'circularity': 0.5, 'aspect_ratio': 1.0, 'solidity': 0.5}

def analyze_facial_proportions(face_gray):
    """Analyze facial proportions using the rule of thirds"""
    try:
        h, w = face_gray.shape

        # Divide face into thirds vertically
        third_h = h // 3
        upper_third = face_gray[0:third_h, :]
        middle_third = face_gray[third_h:2*third_h, :]
        lower_third = face_gray[2*third_h:, :]

        # Analyze width at different levels
        upper_width = analyze_width_at_level(upper_third)
        middle_width = analyze_width_at_level(middle_third)
        lower_width = analyze_width_at_level(lower_third)

        # Calculate proportional relationships
        total_width = upper_width + middle_width + lower_width
        if total_width > 0:
            upper_ratio = upper_width / total_width
            middle_ratio = middle_width / total_width
            lower_ratio = lower_width / total_width
        else:
            upper_ratio = middle_ratio = lower_ratio = 0.33

        # Calculate width variation (how much the width changes)
        width_variation = np.std([upper_width, middle_width, lower_width]) / np.mean([upper_width, middle_width, lower_width]) if np.mean([upper_width, middle_width, lower_width]) > 0 else 0

        return {
            'upper_ratio': upper_ratio,
            'middle_ratio': middle_ratio,
            'lower_ratio': lower_ratio,
            'width_variation': width_variation
        }

    except Exception as e:
        print(f"Error in proportion analysis: {e}")
        return {'upper_ratio': 0.33, 'middle_ratio': 0.34, 'lower_ratio': 0.33, 'width_variation': 0}

def analyze_width_at_level(region):
    """Analyze effective width at a specific face level"""
    try:
        if region.size == 0:
            return 0

        # Apply edge detection
        edges = cv2.Canny(region, 50, 150)

        # Find the median width (more robust than max width)
        widths = []
        for row in edges:
            edge_pixels = np.where(row > 0)[0]
            if len(edge_pixels) >= 2:  # Need at least 2 edge pixels
                width = edge_pixels[-1] - edge_pixels[0]
                widths.append(width)

        return np.median(widths) if widths else 0

    except Exception as e:
        return 0

def analyze_jawline_shape(face_gray):
    """Analyze jawline characteristics"""
    try:
        h, w = face_gray.shape

        # Focus on lower third of face for jawline
        jaw_region = face_gray[int(h*0.6):, :]

        # Apply edge detection to find jaw contour
        edges = cv2.Canny(jaw_region, 50, 150)

        # Find contours in jaw region
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {'jaw_angle': 90, 'jaw_width_ratio': 0.5}

        # Get the largest contour in jaw region
        jaw_contour = max(contours, key=cv2.contourArea)

        # Calculate jaw angle (how angular vs rounded the jaw is)
        # Fit a polygon to approximate jaw shape
        epsilon = 0.02 * cv2.arcLength(jaw_contour, True)
        approx = cv2.approxPolyDP(jaw_contour, epsilon, True)

        # Calculate jaw width relative to face width
        x, y, jaw_w, jaw_h = cv2.boundingRect(jaw_contour)
        jaw_width_ratio = jaw_w / w if w > 0 else 0.5

        # Estimate jaw angle based on contour shape
        if len(approx) <= 4:
            jaw_angle = 70  # Angular jaw
        elif len(approx) <= 6:
            jaw_angle = 90  # Moderate jaw
        else:
            jaw_angle = 120  # Rounded jaw

        return {
            'jaw_angle': jaw_angle,
            'jaw_width_ratio': jaw_width_ratio
        }

    except Exception as e:
        print(f"Error in jawline analysis: {e}")
        return {'jaw_angle': 90, 'jaw_width_ratio': 0.5}

def analyze_forehead_shape(face_gray):
    """Analyze forehead characteristics"""
    try:
        h, w = face_gray.shape

        # Focus on upper third of face for forehead
        forehead_region = face_gray[:int(h*0.4), :]

        # Apply edge detection
        edges = cv2.Canny(forehead_region, 50, 150)

        # Find contours in forehead region
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {'forehead_width_ratio': 0.5, 'forehead_height_ratio': 0.3}

        # Get the largest contour in forehead region
        forehead_contour = max(contours, key=cv2.contourArea)

        # Calculate forehead dimensions
        x, y, forehead_w, forehead_h = cv2.boundingRect(forehead_contour)
        forehead_width_ratio = forehead_w / w if w > 0 else 0.5
        forehead_height_ratio = forehead_h / h if h > 0 else 0.3

        return {
            'forehead_width_ratio': forehead_width_ratio,
            'forehead_height_ratio': forehead_height_ratio
        }

    except Exception as e:
        print(f"Error in forehead analysis: {e}")
        return {'forehead_width_ratio': 0.5, 'forehead_height_ratio': 0.3}

def classify_face_shape_ml_approach(width_height_ratio, contour_features,
                                  proportion_features, jawline_features, forehead_features):
    """ML-inspired face shape classification using multiple features"""

    # Extract features
    circularity = contour_features['circularity']
    aspect_ratio = contour_features['aspect_ratio']
    solidity = contour_features['solidity']

    upper_ratio = proportion_features['upper_ratio']
    middle_ratio = proportion_features['middle_ratio']
    lower_ratio = proportion_features['lower_ratio']
    width_variation = proportion_features['width_variation']

    jaw_angle = jawline_features['jaw_angle']
    jaw_width_ratio = jawline_features['jaw_width_ratio']

    forehead_width_ratio = forehead_features['forehead_width_ratio']
    forehead_height_ratio = forehead_features['forehead_height_ratio']

    # Decision tree-like classification based on facial geometry research

    # ROUND FACE: High circularity, low width variation, rounded jaw
    round_score = (
        (circularity > 0.65) * 3 +
        (0.9 <= width_height_ratio <= 1.1) * 2 +
        (width_variation < 0.15) * 2 +
        (jaw_angle > 110) * 2 +
        (abs(upper_ratio - lower_ratio) < 0.1) * 1
    )

    # OVAL FACE: Moderate circularity, balanced proportions, longer than wide
    oval_score = (
        (0.75 <= width_height_ratio <= 0.9) * 3 +
        (0.4 <= circularity <= 0.7) * 2 +
        (middle_ratio > upper_ratio and middle_ratio > lower_ratio) * 2 +
        (abs(upper_ratio - lower_ratio) < 0.15) * 2 +
        (80 <= jaw_angle <= 110) * 1
    )

    # SQUARE FACE: Low circularity, angular jaw, similar width/height
    square_score = (
        (0.85 <= width_height_ratio <= 1.05) * 3 +
        (circularity < 0.5) * 2 +
        (jaw_angle < 90) * 3 +
        (abs(upper_ratio - lower_ratio) < 0.1) * 2 +
        (solidity > 0.8) * 1
    )

    # HEART FACE: Wide forehead, narrow jaw
    heart_score = (
        (forehead_width_ratio > jaw_width_ratio + 0.1) * 3 +
        (upper_ratio > lower_ratio + 0.15) * 3 +
        (jaw_width_ratio < 0.45) * 2 +
        (forehead_width_ratio > 0.55) * 2 +
        (width_height_ratio <= 0.95) * 1
    )

    # DIAMOND FACE: Wide cheekbones, narrow forehead and jaw
    diamond_score = (
        (middle_ratio > upper_ratio + 0.1 and middle_ratio > lower_ratio + 0.1) * 3 +
        (forehead_width_ratio < 0.5 and jaw_width_ratio < 0.5) * 3 +
        (width_variation > 0.2) * 2 +
        (0.8 <= width_height_ratio <= 1.0) * 1
    )

    # Find the highest scoring face shape
    scores = {
        'Round': round_score,
        'Oval': oval_score,
        'Square': square_score,
        'Heart': heart_score,
        'Diamond': diamond_score
    }

    # Get the face shape with highest score
    best_shape = max(scores, key=scores.get)
    best_score = scores[best_shape]

    # If no clear winner (score < 3), fall back to ratio-based classification
    if best_score < 3:
        return classify_by_ratio_improved(width_height_ratio)

    return best_shape

def classify_by_ratio_improved(ratio):
    """Improved ratio-based classification with better thresholds"""
    if ratio >= 1.05:
        return "Round"
    elif ratio <= 0.8:
        return "Oval"
    elif 0.9 <= ratio < 1.05:
        return "Square"
    elif 0.8 < ratio < 0.9:
        return "Heart"
    else:
        return "Oval"  # Default fallback

# Old functions removed - using new comprehensive analysis

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

@app.route('/classic')
def classic_index():
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
        user = User(username=username, email=email, style_analysis_complete=False)
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
    # Get user statistics
    user_stats = {
        'total_analyses': 5,
        'color_analyses': 3,
        'outfit_recommendations': 12,
        'style_score': 87
    }

    # Get recent analysis (mock data for now)
    recent_analysis = None

    # Get style profile (mock data for now)
    style_profile = {
        'primary_style': 'Modern Classic',
        'color_season': 'Autumn',
        'body_type': 'Athletic',
        'lifestyle': 'Professional'
    }

    return render_template('dashboard.html',
                         user=current_user,
                         user_stats=user_stats,
                         recent_analysis=recent_analysis,
                         style_profile=style_profile)

@app.route('/classic-dashboard')
@login_required
def classic_dashboard():
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

@app.route('/capture_photo')
@login_required
def capture_photo():
    return render_template('image_capture.html')

@app.route('/image_capture')
@login_required
def image_capture():
    return redirect(url_for('capture_photo'))

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

@app.route('/fashion_questionnaire')
@login_required
def fashion_questionnaire():
    """Fashion preference questionnaire"""
    print("=== FASHION QUESTIONNAIRE ACCESSED ===")
    return render_template('fashion_questionnaire.html', user=current_user)

@app.route('/test_route')
def test_route():
    """Simple test route"""
    print("=== TEST ROUTE ACCESSED ===")
    return "Test route is working!"

@app.route('/test_submit')
def test_submit():
    """Test submit route without login"""
    print("=== TEST SUBMIT ROUTE ACCESSED ===")
    return "Submit route is accessible!"

@app.route('/submit_fashion_preferences', methods=['GET', 'POST'])
@login_required
def submit_fashion_preferences():
    """Process fashion preference questionnaire responses"""
    print(f"=== FORM SUBMISSION RECEIVED === Method: {request.method}")

    # Handle GET request (for testing)
    if request.method == 'GET':
        print("GET request received - redirecting to questionnaire")
        flash('Please fill out the questionnaire first.', 'info')
        return redirect(url_for('fashion_questionnaire'))

    try:
        print("Processing fashion preferences submission...")

        # Collect all form data
        preferences = {
            'gender': request.form.get('gender'),
            'age_range': request.form.get('age_range'),
            'lifestyle': request.form.get('lifestyle'),
            'work_environment': request.form.get('work_environment'),
            'social_activities': request.form.getlist('social_activities'),
            'style_inspiration': request.form.getlist('style_inspiration'),
            'preferred_colors': request.form.getlist('preferred_colors'),
            'avoided_colors': request.form.getlist('avoided_colors'),
            'clothing_fit': request.form.get('clothing_fit'),
            'pattern_preference': request.form.getlist('pattern_preference'),
            'fabric_preference': request.form.getlist('fabric_preference'),
            'shopping_frequency': request.form.get('shopping_frequency'),
            'budget_per_item': request.form.get('budget_per_item'),
            'brand_preference': request.form.get('brand_preference'),
            'sustainability_importance': request.form.get('sustainability_importance'),
            'comfort_vs_style': request.form.get('comfort_vs_style'),
            'seasonal_preference': request.form.get('seasonal_preference'),
            'body_concerns': request.form.getlist('body_concerns'),
            'style_goals': request.form.getlist('style_goals'),
            'fashion_risk': request.form.get('fashion_risk'),
            'accessory_preference': request.form.getlist('accessory_preference'),
            'shoe_preference': request.form.getlist('shoe_preference')
        }

        print(f"Collected preferences: {preferences}")

        # Validate required fields
        required_fields = ['gender', 'lifestyle', 'work_environment', 'clothing_fit', 'budget_per_item', 'fashion_risk']
        missing_fields = [field for field in required_fields if not preferences.get(field)]

        if missing_fields:
            flash(f'Please fill in all required fields: {", ".join(missing_fields)}', 'error')
            return redirect(url_for('fashion_questionnaire'))

        # Store preferences in user profile
        try:
            current_user.fashion_preferences = json.dumps(preferences)
            current_user.style_analysis_complete = True

            # Also update user's profile with gender information
            if preferences.get('gender'):
                current_user.gender = preferences['gender'].title()  # Convert to title case for consistency

            db.session.commit()
            print("Preferences saved to database successfully")
            print(f"User {current_user.username} - style_analysis_complete: {current_user.style_analysis_complete}")
            print(f"Fashion preferences length: {len(current_user.fashion_preferences) if current_user.fashion_preferences else 0}")

            # Test analysis generation
            try:
                analysis_results = analyze_fashion_preferences(preferences, current_user)
                print(f"Analysis results generated: {analysis_results}")
            except Exception as analysis_error:
                print(f"Error in analysis generation: {analysis_error}")
                import traceback
                traceback.print_exc()
                # Continue anyway, analysis will be done on recommendations page

            flash('Style preferences saved successfully! Generating your personalized recommendations...', 'success')
            print("Redirecting to force_recommendations...")
            return redirect(url_for('force_recommendations'))

        except Exception as save_error:
            print(f"Error saving preferences to database: {save_error}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            flash('Error saving preferences. Please try again.', 'error')
            return redirect(url_for('fashion_questionnaire'))

    except Exception as e:
        print(f"Error processing fashion preferences: {e}")
        import traceback
        traceback.print_exc()
        flash('Error processing preferences. Please try again.', 'error')
        return redirect(url_for('fashion_questionnaire'))

@app.route('/debug_user')
@login_required
def debug_user():
    """Debug endpoint to check current user state"""
    try:
        db.session.refresh(current_user)
        user_info = {
            'id': current_user.id,
            'username': current_user.username,
            'gender': current_user.gender,
            'style_analysis_complete': current_user.style_analysis_complete,
            'fashion_preferences_length': len(current_user.fashion_preferences) if current_user.fashion_preferences else 0,
            'fashion_preferences_preview': current_user.fashion_preferences[:100] if current_user.fashion_preferences else None
        }
        return jsonify(user_info)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/force_recommendations')
@login_required
def force_recommendations():
    """Force generate recommendations for current user - bypass all checks"""
    try:
        db.session.refresh(current_user)

        if not current_user.fashion_preferences:
            return jsonify({'error': 'No fashion preferences found'})

        preferences = json.loads(current_user.fashion_preferences)
        recommendations = generate_style_recommendations_with_links(preferences, current_user)

        if recommendations and recommendations.get('outfits'):
            return render_template('style_recommendations.html',
                                 recommendations=recommendations,
                                 preferences=preferences,
                                 user=current_user)
        else:
            return jsonify({'error': 'No recommendations generated'})

    except Exception as e:
        import traceback
        error_details = {
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        return jsonify(error_details)

@app.route('/style_recommendations')
@app.route('/style_recommendations/<analysis_id>')
@login_required
def style_recommendations(analysis_id=None):
    """Display personalized style recommendations with shopping links"""
    try:
        print(f"=== STYLE RECOMMENDATIONS REQUEST ===")
        print(f"User: {current_user.username}")

        # Refresh user data from database to ensure we have latest state
        db.session.refresh(current_user)

        print(f"Style analysis complete: {current_user.style_analysis_complete}")
        print(f"Fashion preferences exist: {bool(current_user.fashion_preferences)}")

        # Check if user has completed the style analysis
        if not current_user.style_analysis_complete:
            print(f"User {current_user.username} - style_analysis_complete is False")
            flash('Complete our Style Quiz to get personalized outfit recommendations with shopping links!', 'info')
            return redirect(url_for('fashion_questionnaire'))

        if not current_user.fashion_preferences:
            print(f"User {current_user.username} - fashion_preferences is empty")
            flash('Complete our Style Quiz to get personalized outfit recommendations with shopping links!', 'info')
            return redirect(url_for('fashion_questionnaire'))

        # Get user preferences
        try:
            preferences = json.loads(current_user.fashion_preferences)
            print(f"Loaded preferences: {list(preferences.keys())}")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error loading preferences: {e}")
            flash('There was an issue with your saved preferences. Please retake the Style Quiz.', 'warning')
            return redirect(url_for('fashion_questionnaire'))

        # Generate comprehensive recommendations
        print("Generating style recommendations...")
        try:
            recommendations = generate_style_recommendations_with_links(preferences, current_user)
            print(f"Generated recommendations with {len(recommendations.get('outfits', {}))} outfit categories")

            # Debug: Print recommendation structure
            if recommendations:
                print(f"Recommendation keys: {list(recommendations.keys())}")
                if 'outfits' in recommendations:
                    for category, outfits in recommendations['outfits'].items():
                        print(f"  {category}: {len(outfits)} outfits")

        except Exception as rec_error:
            print(f"Error in recommendation generation: {rec_error}")
            import traceback
            traceback.print_exc()
            flash('Unable to generate recommendations. Please try retaking the Style Quiz.', 'error')
            return redirect(url_for('fashion_questionnaire'))

        # Check if recommendations were generated successfully
        if not recommendations or not recommendations.get('outfits'):
            print("No recommendations generated or empty outfits")
            flash('Unable to generate recommendations. Please retake the Style Quiz.', 'warning')
            return redirect(url_for('fashion_questionnaire'))

        print("Successfully generated recommendations - rendering template")
        return render_template('style_recommendations.html',
                             recommendations=recommendations,
                             preferences=preferences,
                             user=current_user)

    except Exception as e:
        print(f"Error generating style recommendations: {e}")
        import traceback
        traceback.print_exc()
        flash('Unable to generate recommendations at this time. Please try the Style Quiz first.', 'error')
        return redirect(url_for('fashion_questionnaire'))

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
    color_recommendations = generate_color_recommendations(latest_face_analysis, current_user)

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

    color_recommendations = generate_color_recommendations(latest_face_analysis, current_user)

    return render_template('color_analysis.html',
                         color_recommendations=color_recommendations,
                         face_analysis=latest_face_analysis,
                         user=current_user)

def generate_face_shape_recommendations(face_analysis, user=None):
    """Generate personalized recommendations based on face shape and profile"""
    recommendations = {
        'hairstyles': [],
        'glasses': [],
        'accessories': [],
        'makeup_tips': [],
        'clothing_necklines': []
    }

    # Get user's gender for personalized recommendations
    user_gender = None
    if user and hasattr(user, 'gender'):
        user_gender = user.gender.lower() if user.gender else None

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

    # Gender-specific accessory recommendations based on face shape
    if user_gender == 'female':
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
    elif user_gender == 'male':
        accessory_recommendations = {
            'round': [
                {'name': 'Angular Watch', 'description': 'Add definition with geometric timepiece'},
                {'name': 'Structured Tie', 'description': 'Sharp patterns create angles'},
                {'name': 'Square Sunglasses', 'description': 'Add structure to round features'}
            ],
            'oval': [
                {'name': 'Classic Watch', 'description': 'Timeless pieces complement balanced features'},
                {'name': 'Statement Tie', 'description': 'Bold patterns highlight proportions'},
                {'name': 'Aviator Sunglasses', 'description': 'Classic style works with oval faces'}
            ],
            'square': [
                {'name': 'Round Watch', 'description': 'Soft curves balance angular jaw'},
                {'name': 'Textured Tie', 'description': 'Create visual interest and softness'},
                {'name': 'Round Sunglasses', 'description': 'Soft frames complement strong features'}
            ],
            'heart': [
                {'name': 'Bold Watch', 'description': 'Add width at the wrist to balance proportions'},
                {'name': 'Wide Tie', 'description': 'Draw attention to the lower face'},
                {'name': 'Wide-Frame Sunglasses', 'description': 'Balance a wider forehead'}
            ],
            'diamond': [
                {'name': 'Simple Watch', 'description': 'Don\'t compete with strong cheekbones'},
                {'name': 'Narrow Tie', 'description': 'Keep focus on facial features'},
                {'name': 'Minimal Sunglasses', 'description': 'Add width to forehead and jaw areas'}
            ]
        }
    else:
        # Gender-neutral accessories
        accessory_recommendations = {
            'round': [
                {'name': 'Angular Sunglasses', 'description': 'Add definition with geometric frames'},
                {'name': 'Structured Hat', 'description': 'Sharp lines create angles'},
                {'name': 'Geometric Watch', 'description': 'Add structure to round features'}
            ],
            'oval': [
                {'name': 'Classic Sunglasses', 'description': 'Timeless styles complement balanced features'},
                {'name': 'Statement Hat', 'description': 'Bold pieces highlight proportions'},
                {'name': 'Versatile Watch', 'description': 'Works with most styles'}
            ],
            'square': [
                {'name': 'Round Sunglasses', 'description': 'Soft curves balance angular jaw'},
                {'name': 'Soft Hat', 'description': 'Create visual interest and softness'},
                {'name': 'Curved Watch', 'description': 'Soft design complements strong features'}
            ],
            'heart': [
                {'name': 'Wide Sunglasses', 'description': 'Add width to balance proportions'},
                {'name': 'Broad Hat', 'description': 'Draw attention to the lower face'},
                {'name': 'Bold Watch', 'description': 'Balance a wider forehead'}
            ],
            'diamond': [
                {'name': 'Simple Sunglasses', 'description': 'Don\'t compete with strong cheekbones'},
                {'name': 'Minimal Hat', 'description': 'Keep focus on facial features'},
                {'name': 'Clean Watch', 'description': 'Add width to forehead and jaw areas'}
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

    # Add gender-specific makeup/grooming tips based on face shape
    if user_gender == 'female':
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
    elif user_gender == 'male':
        grooming_tips = {
            'round': [
                'Consider angular beard styles to add definition',
                'Keep facial hair well-trimmed and structured',
                'Use a good moisturizer to maintain healthy skin',
                'Consider a hairstyle with height to elongate your face'
            ],
            'oval': [
                'Most beard styles will complement your balanced features',
                'Maintain good skincare routine with cleanser and moisturizer',
                'You can experiment with different facial hair lengths',
                'Keep eyebrows well-groomed but natural'
            ],
            'square': [
                'Soften your strong jawline with a rounded beard style',
                'Use beard oil to keep facial hair healthy',
                'Consider a skincare routine with gentle exfoliation',
                'Keep sideburns neat and proportional'
            ],
            'heart': [
                'Balance your wider forehead with fuller facial hair on the chin',
                'Keep the upper face area clean and well-groomed',
                'Use a good face wash to prevent oiliness in the T-zone',
                'Consider beard styles that add width to the lower face'
            ],
            'diamond': [
                'Add width to forehead and chin areas with appropriate styling',
                'Keep cheekbone area clean to soften prominent features',
                'Use a balanced skincare routine for all face areas',
                'Consider facial hair that adds fullness to the chin'
            ]
        }
        recommendations['makeup_tips'] = grooming_tips.get(face_shape, [])
    else:
        # Gender-neutral grooming tips
        general_tips = {
            'round': [
                'Use skincare products to maintain healthy skin',
                'Keep eyebrows well-groomed',
                'Consider hairstyles that add height',
                'Use sunscreen daily for skin protection'
            ],
            'oval': [
                'Maintain a consistent skincare routine',
                'Your balanced features work with most styles',
                'Keep facial hair (if any) well-maintained',
                'Use moisturizer appropriate for your skin type'
            ],
            'square': [
                'Use gentle skincare products for sensitive areas',
                'Keep any facial hair soft and well-groomed',
                'Consider styles that soften angular features',
                'Use products that maintain skin elasticity'
            ],
            'heart': [
                'Focus on balancing your facial proportions',
                'Use skincare products suitable for combination skin',
                'Keep the forehead area clean and fresh',
                'Maintain overall facial symmetry with grooming'
            ],
            'diamond': [
                'Use products that enhance your natural features',
                'Keep all facial areas well-moisturized',
                'Consider styles that balance your face shape',
                'Maintain consistent grooming routine'
            ]
        }
        recommendations['makeup_tips'] = general_tips.get(face_shape, [])

    return recommendations

def generate_color_recommendations(face_analysis, user=None):
    """Generate gender-specific color recommendations based on skin tone"""
    color_recommendations = {
        'best_colors': [],
        'avoid_colors': [],
        'neutral_colors': [],
        'accent_colors': [],
        'seasonal_palette': '',
        'makeup_colors': {}
    }

    # Get user's gender for personalized recommendations
    user_gender = None
    if user and hasattr(user, 'gender'):
        user_gender = user.gender.lower() if user.gender else None

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
        # Gender-specific makeup/grooming colors
        if user_gender == 'female':
            color_recommendations['makeup_colors'] = palette['makeup_colors']
        elif user_gender == 'male':
            # Male grooming color recommendations
            male_grooming_colors = {
                'beard_care': ['Natural Brown', 'Dark Brown', 'Black'],
                'skincare': ['Clear', 'Natural', 'Unscented'],
                'hair_products': ['Natural', 'Clear', 'Matte']
            }
            color_recommendations['makeup_colors'] = male_grooming_colors
        else:
            # Gender-neutral grooming recommendations
            neutral_colors = {
                'skincare': ['Clear', 'Natural', 'Unscented'],
                'hair_care': ['Natural', 'Clear'],
                'general': ['Neutral tones', 'Natural colors']
            }
            color_recommendations['makeup_colors'] = neutral_colors

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

def analyze_fashion_preferences(preferences, user):
    """Analyze user fashion preferences and create style profile"""
    try:
        # Create comprehensive style profile
        style_profile = {
            'lifestyle_category': categorize_lifestyle(preferences),
            'style_personality': determine_style_personality(preferences),
            'color_palette': analyze_color_preferences(preferences),
            'fit_preferences': analyze_fit_preferences(preferences),
            'budget_category': categorize_budget(preferences),
            'sustainability_score': calculate_sustainability_score(preferences),
            'risk_tolerance': assess_fashion_risk_tolerance(preferences),
            'body_goals': analyze_body_goals(preferences)
        }

        return style_profile

    except Exception as e:
        print(f"Error in fashion preference analysis: {e}")
        return {}

def categorize_lifestyle(preferences):
    """Categorize user lifestyle for targeted recommendations"""
    lifestyle = preferences.get('lifestyle', '')
    work_env = preferences.get('work_environment', '')
    social_activities = preferences.get('social_activities', [])

    # Professional-focused lifestyle
    if work_env in ['corporate', 'business'] or 'networking' in social_activities:
        return 'professional'
    # Active lifestyle
    elif lifestyle == 'active' or 'fitness' in social_activities:
        return 'active'
    # Creative lifestyle
    elif work_env == 'creative' or 'art_events' in social_activities:
        return 'creative'
    # Casual lifestyle
    elif lifestyle == 'casual' or 'casual_hangouts' in social_activities:
        return 'casual'
    # Social lifestyle
    elif 'parties' in social_activities or 'dining' in social_activities:
        return 'social'
    else:
        return 'balanced'

def determine_style_personality(preferences):
    """Determine user's style personality type"""
    style_inspiration = preferences.get('style_inspiration', [])
    fashion_risk = preferences.get('fashion_risk', 'moderate')
    comfort_vs_style = preferences.get('comfort_vs_style', 'balanced')

    # Classic style
    if 'classic' in style_inspiration and fashion_risk == 'conservative':
        return 'classic'
    # Trendy style
    elif 'trendy' in style_inspiration and fashion_risk == 'adventurous':
        return 'trendy'
    # Bohemian style
    elif 'bohemian' in style_inspiration:
        return 'bohemian'
    # Minimalist style
    elif 'minimalist' in style_inspiration:
        return 'minimalist'
    # Edgy style
    elif 'edgy' in style_inspiration and fashion_risk == 'adventurous':
        return 'edgy'
    # Romantic style
    elif 'romantic' in style_inspiration:
        return 'romantic'
    # Comfort-focused
    elif comfort_vs_style == 'comfort':
        return 'comfort'
    else:
        return 'eclectic'

def analyze_color_preferences(preferences):
    """Analyze color preferences and create palette"""
    preferred = preferences.get('preferred_colors', [])
    avoided = preferences.get('avoided_colors', [])

    # Create color categories
    warm_colors = ['red', 'orange', 'yellow', 'coral', 'peach']
    cool_colors = ['blue', 'green', 'purple', 'teal', 'mint']
    neutral_colors = ['black', 'white', 'gray', 'beige', 'brown', 'navy']

    # Analyze preferences
    warm_score = sum(1 for color in preferred if color in warm_colors)
    cool_score = sum(1 for color in preferred if color in cool_colors)
    neutral_score = sum(1 for color in preferred if color in neutral_colors)

    # Determine dominant palette
    if warm_score > cool_score and warm_score > neutral_score:
        palette_type = 'warm'
    elif cool_score > warm_score and cool_score > neutral_score:
        palette_type = 'cool'
    elif neutral_score > warm_score and neutral_score > cool_score:
        palette_type = 'neutral'
    else:
        palette_type = 'mixed'

    return {
        'type': palette_type,
        'preferred': preferred,
        'avoided': avoided,
        'warm_score': warm_score,
        'cool_score': cool_score,
        'neutral_score': neutral_score
    }

def analyze_fit_preferences(preferences):
    """Analyze clothing fit preferences"""
    fit = preferences.get('clothing_fit', 'regular')
    body_concerns = preferences.get('body_concerns', [])

    fit_profile = {
        'primary_fit': fit,
        'body_concerns': body_concerns,
        'recommendations': []
    }

    # Add fit recommendations based on concerns
    if 'hide_midsection' in body_concerns:
        fit_profile['recommendations'].extend(['empire_waist', 'a_line', 'wrap_style'])
    if 'enhance_curves' in body_concerns:
        fit_profile['recommendations'].extend(['fitted', 'bodycon', 'wrap_style'])
    if 'elongate_legs' in body_concerns:
        fit_profile['recommendations'].extend(['high_waisted', 'cropped_tops', 'vertical_lines'])
    if 'broaden_shoulders' in body_concerns:
        fit_profile['recommendations'].extend(['structured_shoulders', 'horizontal_stripes', 'boat_neck'])

    return fit_profile

def categorize_budget(preferences):
    """Categorize budget for appropriate recommendations"""
    budget = preferences.get('budget_per_item', '')

    if budget == 'under_25':
        return 'budget'
    elif budget == '25_50':
        return 'affordable'
    elif budget == '50_100':
        return 'moderate'
    elif budget == '100_200':
        return 'premium'
    elif budget == 'over_200':
        return 'luxury'
    else:
        return 'moderate'

def calculate_sustainability_score(preferences):
    """Calculate sustainability importance score"""
    sustainability = preferences.get('sustainability_importance', 'somewhat')

    if sustainability == 'very_important':
        return 5
    elif sustainability == 'important':
        return 4
    elif sustainability == 'somewhat':
        return 3
    elif sustainability == 'not_much':
        return 2
    else:
        return 1

def assess_fashion_risk_tolerance(preferences):
    """Assess user's fashion risk tolerance"""
    risk = preferences.get('fashion_risk', 'moderate')
    style_goals = preferences.get('style_goals', [])

    risk_score = 3  # Default moderate

    if risk == 'conservative':
        risk_score = 1
    elif risk == 'adventurous':
        risk_score = 5

    # Adjust based on style goals
    if 'experiment_new_styles' in style_goals:
        risk_score += 1
    if 'build_classic_wardrobe' in style_goals:
        risk_score -= 1

    return max(1, min(5, risk_score))

def analyze_body_goals(preferences):
    """Analyze body-related styling goals"""
    concerns = preferences.get('body_concerns', [])
    goals = preferences.get('style_goals', [])

    return {
        'concerns': concerns,
        'goals': goals,
        'focus_areas': list(set(concerns + [g for g in goals if 'body' in g or 'figure' in g]))
    }

def generate_style_recommendations_with_links(preferences, user):
    """Generate comprehensive style recommendations with shopping links"""
    try:
        # Analyze preferences to create style profile
        style_profile = analyze_fashion_preferences(preferences, user)

        # Generate outfit recommendations
        outfit_recommendations = generate_outfit_recommendations(style_profile, preferences, user)

        # Add shopping links for each recommendation
        recommendations_with_links = add_shopping_links(outfit_recommendations, style_profile)

        return {
            'style_profile': style_profile,
            'outfits': recommendations_with_links,
            'shopping_categories': generate_shopping_categories(style_profile),
            'seasonal_recommendations': generate_seasonal_recommendations(style_profile, preferences),
            'accessory_recommendations': generate_accessory_recommendations(style_profile, preferences, user)
        }

    except Exception as e:
        print(f"Error generating style recommendations: {e}")
        return {}

def get_outfit_images():
    """Professional outfit images showcasing specific clothing pieces for board presentation"""
    return {
        # Female outfit images - Professional styling showcasing actual clothing pieces
        'female': {
            # Work Outfits - Professional business attire with clear clothing visibility
            'blazer_outfit': 'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
            'business_casual': 'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
            'professional_dress': 'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',

            # Casual Outfits - Stylish everyday wear showcasing outfit pieces
            'casual_chic': 'https://images.unsplash.com/photo-1469334031218-e382a71b716b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
            'weekend_comfort': 'https://images.unsplash.com/photo-1445205170230-053b83016050?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',

            # Evening Outfits - Elegant evening wear with clear garment details
            'little_black_dress': 'https://images.unsplash.com/photo-1566479179817-c0b5b4b4b1e5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
            'cocktail_dress': 'https://images.unsplash.com/photo-1595777457583-95e059d581b8?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',

            # Style-specific outfits - Unique styling with visible clothing elements
            'bohemian_style': 'https://images.unsplash.com/photo-1509631179647-0177331693ae?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
            'edgy_style': 'https://images.unsplash.com/photo-1558769132-cb1aea458c5e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
        },

        # Male outfit images - Professional styling showcasing specific clothing pieces
        'male': {
            # Work Outfits - Business attire with clear visibility of suit components
            'blazer_outfit': 'https://images.unsplash.com/photo-1617137984095-74e4e5e3613f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
            'business_casual': 'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
            'professional_dress': 'https://images.unsplash.com/photo-1617137984095-74e4e5e3613f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',

            # Casual Outfits - Smart casual wear showcasing individual pieces
            'casual_chic': 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
            'weekend_comfort': 'https://images.unsplash.com/photo-1564564321837-a57b7070ac4f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',

            # Evening Outfits - Formal evening wear with clear garment details
            'date_night_smart': 'https://images.unsplash.com/photo-1617137984095-74e4e5e3613f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
            'formal_evening': 'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',

            # Style-specific outfits - Creative styling with visible clothing elements
            'bohemian_style': 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
            'edgy_style': 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
        }
    }

def get_individual_item_images():
    """
    Professional images for individual clothing pieces.
    Each item has multiple high-quality images focusing on the garment, not the person.
    """
    return {
        'male': {
            'suit_jacket': [
                'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1617137984095-74e4e5e3613f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'dress_pants': [
                'https://images.unsplash.com/photo-1473966968600-fa801b869a1a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1506629905607-d405b7a30db9?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1441986300917-64674bd600d8?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'dress_shirt': [
                'https://images.unsplash.com/photo-1602810318383-e386cc2a3ccf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1586790170083-2f9ceadc732d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1621072156002-e2fccdc0b176?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'tie': [
                'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'dress_shoes': [
                'https://images.unsplash.com/photo-1549298916-b41d501d3772?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1582897085656-c636d006a246?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1614252369475-531eba835eb1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'briefcase': [
                'https://images.unsplash.com/photo-1584464491033-06628f3a6b7b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1590736969955-71cc94901144?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'blazer': [
                'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1617137984095-74e4e5e3613f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'chinos': [
                'https://images.unsplash.com/photo-1506629905607-d405b7a30db9?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1441986300917-64674bd600d8?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1473966968600-fa801b869a1a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'button_down_shirt': [
                'https://images.unsplash.com/photo-1586790170083-2f9ceadc732d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1621072156002-e2fccdc0b176?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1602810318383-e386cc2a3ccf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'loafers': [
                'https://images.unsplash.com/photo-1582897085656-c636d006a246?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1614252369475-531eba835eb1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1549298916-b41d501d3772?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'leather_bag': [
                'https://images.unsplash.com/photo-1590736969955-71cc94901144?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1584464491033-06628f3a6b7b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'polo_shirt': [
                'https://images.unsplash.com/photo-1586790170083-2f9ceadc732d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1621072156002-e2fccdc0b176?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1602810318383-e386cc2a3ccf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'watch': [
                'https://images.unsplash.com/photo-1524592094714-0f0654e20314?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1522312346375-d1a52e2b99b3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1434056886845-dac89ffe9b56?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'suit': [
                'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1617137984095-74e4e5e3613f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            # Additional male items for outfit recommendations
            'dark_jeans': [
                'https://images.unsplash.com/photo-1542272604-787c3835535d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1475178626620-a4d074967452?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1541099649105-f69ad21f3246?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'blazer': [
                'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1617137984095-74e4e5e3613f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'cufflinks': [
                'https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1506630448388-4e683c67ddb0?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1611652022419-a9419f74343d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'cufflinks': [
                'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1617137984095-74e4e5e3613f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'dark_jeans': [
                'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1617137984095-74e4e5e3613f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ]
        },
        'female': {
            'blazer': [
                'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1617137984095-74e4e5e3613f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'dress_pants': [
                'https://images.unsplash.com/photo-1473966968600-fa801b869a1a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1506629905607-d405b7a30db9?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1441986300917-64674bd600d8?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'blouse': [
                'https://images.unsplash.com/photo-1602810318383-e386cc2a3ccf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1586790170083-2f9ceadc732d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1621072156002-e2fccdc0b176?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'pumps': [
                'https://images.unsplash.com/photo-1549298916-b41d501d3772?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1582897085656-c636d006a246?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1614252369475-531eba835eb1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'structured_bag': [
                'https://images.unsplash.com/photo-1584464491033-06628f3a6b7b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1590736969955-71cc94901144?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'little_black_dress': [
                'https://images.unsplash.com/photo-1566479179817-c0b5b4b4b1e5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1595777457583-95e059d581b8?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'heels': [
                'https://images.unsplash.com/photo-1549298916-b41d501d3772?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1582897085656-c636d006a246?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1614252369475-531eba835eb1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'clutch': [
                'https://images.unsplash.com/photo-1584464491033-06628f3a6b7b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1590736969955-71cc94901144?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'statement_earrings': [
                'https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1506630448388-4e683c67ddb0?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1611652022419-a9419f74343d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'cocktail_dress': [
                'https://images.unsplash.com/photo-1595777457583-95e059d581b8?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1566479179817-c0b5b4b4b1e5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'strappy_heels': [
                'https://images.unsplash.com/photo-1549298916-b41d501d3772?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1582897085656-c636d006a246?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1614252369475-531eba835eb1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'evening_bag': [
                'https://images.unsplash.com/photo-1584464491033-06628f3a6b7b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1590736969955-71cc94901144?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'bold_jewelry': [
                'https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1506630448388-4e683c67ddb0?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1611652022419-a9419f74343d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            # Additional items for work outfits
            'trousers': [
                'https://images.unsplash.com/photo-1473966968600-fa801b869a1a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1506629905607-d405b7a30db9?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1441986300917-64674bd600d8?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'cardigan': [
                'https://images.unsplash.com/photo-1602810318383-e386cc2a3ccf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1586790170083-2f9ceadc732d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1621072156002-e2fccdc0b176?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'loafers': [
                'https://images.unsplash.com/photo-1582897085656-c636d006a246?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1614252369475-531eba835eb1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1549298916-b41d501d3772?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'tote_bag': [
                'https://images.unsplash.com/photo-1584464491033-06628f3a6b7b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1590736969955-71cc94901144?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'sheath_dress': [
                'https://images.unsplash.com/photo-1566479179817-c0b5b4b4b1e5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1595777457583-95e059d581b8?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ],
            'statement_necklace': [
                'https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1506630448388-4e683c67ddb0?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1611652022419-a9419f74343d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ]
        }
    }

def get_professional_outfit_images():
    """
    Professional outfit images specifically curated for board presentation.
    Each image clearly showcases the clothing pieces mentioned in outfit recommendations.
    All images are unique and professionally styled to show specific garments.
    """
    return {
        'male': {
            # Date Night Smart - Dress shirt, dark jeans, blazer, dress shoes, watch
            'date_night_smart': 'https://images.unsplash.com/photo-1617137984095-74e4e5e3613f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',

            # Formal Evening - Suit, dress shirt, tie, dress shoes, cufflinks
            'formal_evening': 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',

            # Business Professional - Suit, dress shirt, tie
            'business_professional': 'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',

            # Smart Casual - Chinos, polo, loafers
            'smart_casual': 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',

            # Blazer Outfit - Blazer, chinos, button-down
            'blazer_outfit': 'https://images.unsplash.com/photo-1556157382-97eda2d62296?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
        },
        'female': {
            # Date Night Chic - Little black dress, heels, clutch, statement earrings
            'date_night_chic': 'https://images.unsplash.com/photo-1566479179817-c0b5b4b4b1e5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',

            # Night Out Glam - Cocktail dress, strappy heels, evening bag, bold jewelry
            'night_out_glam': 'https://images.unsplash.com/photo-1595777457583-95e059d581b8?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',

            # Business Professional - Blazer, trousers, blouse
            'business_professional': 'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',

            # Smart Casual - Midi dress, denim jacket, sneakers
            'smart_casual': 'https://images.unsplash.com/photo-1469334031218-e382a71b716b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',

            # Blazer Outfit - Blazer, trousers, blouse
            'blazer_outfit': 'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
        }
    }

def get_outfit_specific_image(outfit_name, gender, items):
    """
    Get a specific professional image for each outfit that showcases the mentioned clothing pieces.
    This ensures every outfit has a unique, relevant image for board presentation.
    """
    professional_images = get_professional_outfit_images()

    # Map outfit names to specific images
    outfit_image_mapping = {
        'male': {
            'Date Night Smart': professional_images['male']['date_night_smart'],
            'Formal Evening': professional_images['male']['formal_evening'],
            'Classic Business Suit': professional_images['male']['business_professional'],
            'Smart Casual Office Look': professional_images['male']['blazer_outfit'],
            'Professional Polo Look': professional_images['male']['smart_casual'],
        },
        'female': {
            'Date Night Chic': professional_images['female']['date_night_chic'],
            'Formal Evening': professional_images['female']['night_out_glam'],
            'Classic Business Suit': professional_images['female']['business_professional'],
            'Smart Casual Office Look': professional_images['female']['blazer_outfit'],
            'Dress and Blazer Combo': professional_images['female']['smart_casual'],
        }
    }

    # Return specific image or fallback to a default professional image
    return outfit_image_mapping.get(gender, {}).get(outfit_name,
        professional_images[gender]['business_professional'] if gender in professional_images else
        'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80')

def get_item_images_for_outfit(items, gender):
    """
    Get multiple images for each clothing item in an outfit.
    Returns a dictionary with item names as keys and lists of images as values.
    """
    item_images_db = get_individual_item_images()
    item_images = {}

    for item in items:
        # Normalize item name (remove spaces, convert to lowercase with underscores)
        item_key = item.lower().replace(' ', '_')

        # Get images for this item
        if gender in item_images_db and item_key in item_images_db[gender]:
            item_images[item] = item_images_db[gender][item_key]
        else:
            # Fallback to default professional images if specific item not found
            item_images[item] = [
                'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1617137984095-74e4e5e3613f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80'
            ]

    return item_images

def generate_outfit_recommendations(style_profile, preferences, user=None):
    """Generate specific outfit recommendations based on style profile and user gender"""

    lifestyle = style_profile.get('lifestyle_category', 'balanced')
    style_personality = style_profile.get('style_personality', 'eclectic')
    color_palette = style_profile.get('color_palette', {})

    # Get user's gender for personalized recommendations
    user_gender = None
    if user and hasattr(user, 'gender'):
        user_gender = user.gender.lower() if user.gender else None

    print(f"DEBUG: User object: {user}")
    print(f"DEBUG: User gender raw: {user.gender if user else 'No user'}")
    print(f"DEBUG: User gender processed: {user_gender}")

    # Default to 'male' if gender not specified or invalid
    if not user_gender or user_gender not in ['male', 'female']:
        user_gender = 'male'

    print(f"DEBUG: Final user_gender: {user_gender}")

    # Get outfit images
    outfit_images = get_outfit_images()

    outfit_categories = {
        'work_outfits': [],
        'casual_outfits': [],
        'evening_outfits': [],
        'weekend_outfits': [],
        'special_occasion': []
    }

    # Work outfits based on lifestyle and gender
    if lifestyle == 'professional':
        if user_gender == 'male':
            outfit_categories['work_outfits'] = [
                {
                    'name': 'Classic Business Suit',
                    'description': 'Tailored suit with dress shirt and tie',
                    'items': ['Suit Jacket', 'Dress Pants', 'Dress Shirt', 'Tie', 'Dress Shoes', 'Briefcase'],
                    'colors': ['Navy', 'Charcoal', 'Black'],
                    'occasion': 'Business Meetings',
                    'formality': 'Formal',
                    'image': get_outfit_specific_image('Classic Business Suit', 'male', ['Suit Jacket', 'Dress Pants', 'Dress Shirt', 'Tie', 'Dress Shoes', 'Briefcase']),
                    'item_images': get_item_images_for_outfit(['Suit Jacket', 'Dress Pants', 'Dress Shirt', 'Tie', 'Dress Shoes', 'Briefcase'], 'male')
                },
                {
                    'name': 'Smart Casual Office Look',
                    'description': 'Blazer with chinos and button-down shirt',
                    'items': ['Blazer', 'Chinos', 'Button Down Shirt', 'Loafers', 'Leather Bag'],
                    'colors': ['Navy', 'Khaki', 'White'],
                    'occasion': 'Office Daily',
                    'formality': 'Business Casual',
                    'image': get_outfit_specific_image('Smart Casual Office Look', 'male', ['Blazer', 'Chinos', 'Button Down Shirt', 'Loafers', 'Leather Bag']),
                    'item_images': get_item_images_for_outfit(['Blazer', 'Chinos', 'Button Down Shirt', 'Loafers', 'Leather Bag'], 'male')
                },
                {
                    'name': 'Professional Polo Look',
                    'description': 'Polo shirt with dress pants and blazer',
                    'items': ['Polo Shirt', 'Dress Pants', 'Blazer', 'Dress Shoes', 'Watch'],
                    'colors': ['Navy', 'White', 'Gray'],
                    'occasion': 'Casual Friday',
                    'formality': 'Smart Casual',
                    'image': get_outfit_specific_image('Professional Polo Look', 'male', ['Polo Shirt', 'Dress Pants', 'Blazer', 'Dress Shoes', 'Watch']),
                    'item_images': get_item_images_for_outfit(['Polo Shirt', 'Dress Pants', 'Blazer', 'Dress Shoes', 'Watch'], 'male')
                }
            ]
        else:  # female
            outfit_categories['work_outfits'] = [
                {
                    'name': 'Classic Business Suit',
                    'description': 'Tailored blazer with matching trousers or pencil skirt',
                    'items': ['Blazer', 'Dress Pants', 'Blouse', 'Pumps', 'Structured Bag'],
                    'colors': ['Navy', 'Charcoal', 'Black'],
                    'occasion': 'Business Meetings',
                    'formality': 'Formal',
                    'image': get_outfit_specific_image('Classic Business Suit', 'female', ['Blazer', 'Dress Pants', 'Blouse', 'Pumps', 'Structured Bag']),
                    'item_images': get_item_images_for_outfit(['Blazer', 'Dress Pants', 'Blouse', 'Pumps', 'Structured Bag'], 'female')
                },
                {
                    'name': 'Smart Casual Office Look',
                    'description': 'Blouse with tailored trousers and cardigan',
                    'items': ['Blouse', 'Trousers', 'Cardigan', 'Loafers', 'Tote Bag'],
                    'colors': ['White', 'Navy', 'Camel'],
                    'occasion': 'Office Daily',
                    'formality': 'Business Casual',
                    'image': get_outfit_specific_image('Smart Casual Office Look', 'female', ['Blouse', 'Trousers', 'Cardigan', 'Loafers', 'Tote Bag']),
                    'item_images': get_item_images_for_outfit(['Blouse', 'Trousers', 'Cardigan', 'Loafers', 'Tote Bag'], 'female')
                },
                {
                    'name': 'Dress and Blazer Combo',
                    'description': 'Sheath dress with structured blazer',
                    'items': ['Sheath Dress', 'Blazer', 'Pumps', 'Statement Necklace'],
                    'colors': ['Black', 'Navy', 'Burgundy'],
                    'occasion': 'Presentations',
                    'formality': 'Business Formal',
                    'image': get_outfit_specific_image('Dress and Blazer Combo', 'female', ['Sheath Dress', 'Blazer', 'Pumps', 'Statement Necklace']),
                    'item_images': get_item_images_for_outfit(['Sheath Dress', 'Blazer', 'Pumps', 'Statement Necklace'], 'female')
                }
            ]
    elif lifestyle == 'creative':
        if user_gender == 'male':
            outfit_categories['work_outfits'] = [
                {
                    'name': 'Artistic Professional',
                    'description': 'Unique blazer with creative accessories',
                    'items': ['printed_blazer', 'dark_jeans', 'graphic_tee', 'chelsea_boots', 'messenger_bag'],
                    'colors': ['jewel_tones', 'black', 'artistic_prints'],
                    'occasion': 'creative_meetings',
                    'formality': 'creative_casual',
                    'image': outfit_images['male']['edgy_style']
                },
                {
                    'name': 'Smart Creative Look',
                    'description': 'Button-down with creative accessories',
                    'items': ['patterned_shirt', 'dark_jeans', 'blazer', 'loafers', 'leather_bag'],
                    'colors': ['earth_tones', 'jewel_colors'],
                    'occasion': 'client_meetings',
                    'formality': 'creative_business',
                    'image': outfit_images['male']['bohemian_style']
                }
            ]
        else:  # female
            outfit_categories['work_outfits'] = [
                {
                    'name': 'Artistic Professional',
                    'description': 'Unique blazer with creative accessories',
                    'items': ['printed_blazer', 'dark_jeans', 'artistic_top', 'ankle_boots', 'creative_bag'],
                    'colors': ['jewel_tones', 'black', 'artistic_prints'],
                    'occasion': 'creative_meetings',
                    'formality': 'creative_casual',
                    'image': outfit_images['female']['edgy_style']
                },
                {
                    'name': 'Bohemian Professional',
                    'description': 'Flowy blouse with structured bottom',
                    'items': ['bohemian_blouse', 'tailored_pants', 'statement_jewelry', 'block_heels'],
                    'colors': ['earth_tones', 'jewel_colors'],
                    'occasion': 'client_meetings',
                    'formality': 'creative_business',
                    'image': outfit_images['female']['bohemian_style']
                }
            ]

    # Casual outfits based on gender
    if user_gender == 'male':
        outfit_categories['casual_outfits'] = [
            {
                'name': 'Weekend Comfort',
                'description': 'Comfortable yet stylish weekend wear',
                'items': ['comfortable_jeans', 'henley_shirt', 'sneakers', 'baseball_cap'],
                'colors': color_palette.get('preferred', ['denim', 'neutral']),
                'occasion': 'weekend_errands',
                'formality': 'casual',
                'image': outfit_images['male']['weekend_comfort']
            },
            {
                'name': 'Smart Casual',
                'description': 'Elevated casual for social outings',
                'items': ['chinos', 'polo_shirt', 'loafers', 'watch'],
                'colors': ['navy', 'khaki', 'white'],
                'occasion': 'brunch_friends',
                'formality': 'smart_casual',
                'image': outfit_images['male']['casual_chic']
            }
        ]
    else:  # female
        outfit_categories['casual_outfits'] = [
            {
                'name': 'Weekend Comfort',
                'description': 'Comfortable yet stylish weekend wear',
                'items': ['comfortable_jeans', 'soft_sweater', 'sneakers', 'crossbody_bag'],
                'colors': color_palette.get('preferred', ['denim', 'neutral']),
                'occasion': 'weekend_errands',
                'formality': 'casual',
                'image': outfit_images['female']['weekend_comfort']
            },
            {
                'name': 'Brunch Ready',
                'description': 'Elevated casual for social outings',
                'items': ['midi_dress', 'denim_jacket', 'white_sneakers', 'small_bag'],
                'colors': ['pastels', 'florals', 'light_colors'],
                'occasion': 'brunch_friends',
                'formality': 'smart_casual',
                'image': outfit_images['female']['casual_chic']
            }
        ]

    # Evening outfits based on gender - Using professional images that showcase clothing pieces
    if user_gender == 'male':
        outfit_categories['evening_outfits'] = [
            {
                'name': 'Date Night Smart',
                'description': 'Elegant yet approachable evening look',
                'items': ['Dress Shirt', 'Dark Jeans', 'Blazer', 'Dress Shoes', 'Watch'],
                'colors': ['Black', 'Navy', 'Burgundy'],
                'occasion': 'Dinner Date',
                'formality': 'Semi-Formal',
                'image': get_outfit_specific_image('Date Night Smart', 'male', ['Dress Shirt', 'Dark Jeans', 'Blazer', 'Dress Shoes', 'Watch']),
                'item_images': get_item_images_for_outfit(['Dress Shirt', 'Dark Jeans', 'Blazer', 'Dress Shoes', 'Watch'], 'male')
            },
            {
                'name': 'Formal Evening',
                'description': 'Sophisticated outfit for special evenings',
                'items': ['Suit', 'Dress Shirt', 'Tie', 'Dress Shoes', 'Cufflinks'],
                'colors': ['Black', 'Navy', 'Charcoal'],
                'occasion': 'Parties Events',
                'formality': 'Formal',
                'image': get_outfit_specific_image('Formal Evening', 'male', ['Suit', 'Dress Shirt', 'Tie', 'Dress Shoes', 'Cufflinks']),
                'item_images': get_item_images_for_outfit(['Suit', 'Dress Shirt', 'Tie', 'Dress Shoes', 'Cufflinks'], 'male')
            }
        ]
    else:  # female
        outfit_categories['evening_outfits'] = [
            {
                'name': 'Date Night Chic',
                'description': 'Elegant yet approachable evening look',
                'items': ['Little Black Dress', 'Heels', 'Clutch', 'Statement Earrings'],
                'colors': ['Black', 'Navy', 'Burgundy'],
                'occasion': 'Dinner Date',
                'formality': 'Semi-Formal',
                'image': get_outfit_specific_image('Date Night Chic', 'female', ['Little Black Dress', 'Heels', 'Clutch', 'Statement Earrings']),
                'item_images': get_item_images_for_outfit(['Little Black Dress', 'Heels', 'Clutch', 'Statement Earrings'], 'female')
            },
            {
                'name': 'Formal Evening',
                'description': 'Sophisticated outfit for special evenings',
                'items': ['Cocktail Dress', 'Strappy Heels', 'Evening Bag', 'Bold Jewelry'],
                'colors': ['Jewel Tones', 'Metallics', 'Black'],
                'occasion': 'Parties Events',
                'formality': 'Formal',
                'image': get_outfit_specific_image('Formal Evening', 'female', ['Cocktail Dress', 'Strappy Heels', 'Evening Bag', 'Bold Jewelry']),
                'item_images': get_item_images_for_outfit(['Cocktail Dress', 'Strappy Heels', 'Evening Bag', 'Bold Jewelry'], 'female')
            }
        ]

    return outfit_categories

def add_shopping_links(outfit_recommendations, style_profile):
    """Add shopping links to outfit recommendations"""

    budget_category = style_profile.get('budget_category', 'moderate')
    sustainability_score = style_profile.get('sustainability_score', 3)

    # Define shopping sources based on budget and sustainability
    shopping_sources = get_shopping_sources(budget_category, sustainability_score)

    # Add links to each outfit
    for category, outfits in outfit_recommendations.items():
        for outfit in outfits:
            outfit['shopping_links'] = generate_item_links(outfit['items'], shopping_sources)
            outfit['total_estimated_cost'] = calculate_outfit_cost(outfit['items'], budget_category)

    return outfit_recommendations

def get_shopping_sources(budget_category, sustainability_score):
    """Get appropriate shopping sources based on budget and sustainability preferences"""

    sources = {
        'budget': [
            {'name': 'H&M', 'url': 'https://www2.hm.com/en_us/index.html', 'type': 'fast_fashion'},
            {'name': 'Target', 'url': 'https://www.target.com/c/clothing/-/N-5xu0o', 'type': 'affordable'},
            {'name': 'Old Navy', 'url': 'https://oldnavy.gap.com/', 'type': 'budget'},
            {'name': 'Forever 21', 'url': 'https://www.forever21.com/', 'type': 'trendy_budget'}
        ],
        'affordable': [
            {'name': 'Zara', 'url': 'https://www.zara.com/us/', 'type': 'trendy'},
            {'name': 'Uniqlo', 'url': 'https://www.uniqlo.com/us/en/', 'type': 'basics'},
            {'name': 'J.Crew Factory', 'url': 'https://factory.jcrew.com/', 'type': 'classic'},
            {'name': 'Banana Republic Factory', 'url': 'https://bananarepublicfactory.gapfactory.com/', 'type': 'professional'}
        ],
        'moderate': [
            {'name': 'J.Crew', 'url': 'https://www.jcrew.com/', 'type': 'classic'},
            {'name': 'Banana Republic', 'url': 'https://bananarepublic.gap.com/', 'type': 'professional'},
            {'name': 'Anthropologie', 'url': 'https://www.anthropologie.com/', 'type': 'bohemian'},
            {'name': 'Nordstrom', 'url': 'https://www.nordstrom.com/', 'type': 'department_store'}
        ],
        'premium': [
            {'name': 'Theory', 'url': 'https://www.theory.com/', 'type': 'minimalist'},
            {'name': 'Equipment', 'url': 'https://www.equipmentfr.com/', 'type': 'luxury_basics'},
            {'name': 'Reformation', 'url': 'https://www.thereformation.com/', 'type': 'sustainable'},
            {'name': 'Everlane', 'url': 'https://www.everlane.com/', 'type': 'ethical'}
        ],
        'luxury': [
            {'name': 'Net-A-Porter', 'url': 'https://www.net-a-porter.com/', 'type': 'luxury'},
            {'name': 'Saks Fifth Avenue', 'url': 'https://www.saksfifthavenue.com/', 'type': 'luxury'},
            {'name': 'Nordstrom', 'url': 'https://www.nordstrom.com/', 'type': 'department_store'},
            {'name': 'Bergdorf Goodman', 'url': 'https://www.bergdorfgoodman.com/', 'type': 'luxury'}
        ]
    }

    # Add sustainable options if sustainability is important
    if sustainability_score >= 4:
        sustainable_sources = [
            {'name': 'Everlane', 'url': 'https://www.everlane.com/', 'type': 'sustainable'},
            {'name': 'Reformation', 'url': 'https://www.thereformation.com/', 'type': 'sustainable'},
            {'name': 'Eileen Fisher', 'url': 'https://www.eileenfisher.com/', 'type': 'sustainable'},
            {'name': 'Patagonia', 'url': 'https://www.patagonia.com/', 'type': 'sustainable_outdoor'}
        ]
        sources[budget_category].extend(sustainable_sources)

    return sources.get(budget_category, sources['moderate'])

def generate_item_links(items, shopping_sources):
    """Generate shopping links for specific clothing items"""

    item_categories = {
        'blazer': 'blazers',
        'dress_pants': 'pants',
        'blouse': 'tops',
        'pumps': 'shoes',
        'structured_bag': 'bags',
        'cardigan': 'sweaters',
        'trousers': 'pants',
        'loafers': 'shoes',
        'tote_bag': 'bags',
        'sheath_dress': 'dresses',
        'statement_necklace': 'jewelry',
        'printed_blazer': 'blazers',
        'dark_jeans': 'jeans',
        'artistic_top': 'tops',
        'ankle_boots': 'shoes',
        'creative_bag': 'bags',
        'bohemian_blouse': 'tops',
        'tailored_pants': 'pants',
        'statement_jewelry': 'jewelry',
        'block_heels': 'shoes',
        'comfortable_jeans': 'jeans',
        'soft_sweater': 'sweaters',
        'sneakers': 'shoes',
        'crossbody_bag': 'bags',
        'midi_dress': 'dresses',
        'denim_jacket': 'jackets',
        'white_sneakers': 'shoes',
        'small_bag': 'bags',
        'little_black_dress': 'dresses',
        'heels': 'shoes',
        'clutch': 'bags',
        'statement_earrings': 'jewelry',
        'cocktail_dress': 'dresses',
        'strappy_heels': 'shoes',
        'evening_bag': 'bags',
        'bold_jewelry': 'jewelry'
    }

    links = []
    for item in items:
        category = item_categories.get(item, 'clothing')
        for source in shopping_sources:
            link = {
                'item': item.replace('_', ' ').title(),
                'store': source['name'],
                'url': f"{source['url']}/search?q={item.replace('_', '+')}&category={category}",
                'type': source['type']
            }
            links.append(link)

    return links

def calculate_outfit_cost(items, budget_category):
    """Calculate estimated outfit cost based on budget category"""

    # Average item costs by budget category
    cost_ranges = {
        'budget': {'top': 15, 'bottom': 20, 'dress': 25, 'shoes': 30, 'bag': 20, 'jewelry': 10, 'outerwear': 35},
        'affordable': {'top': 30, 'bottom': 40, 'dress': 50, 'shoes': 60, 'bag': 40, 'jewelry': 25, 'outerwear': 70},
        'moderate': {'top': 60, 'bottom': 80, 'dress': 100, 'shoes': 120, 'bag': 80, 'jewelry': 50, 'outerwear': 140},
        'premium': {'top': 120, 'bottom': 160, 'dress': 200, 'shoes': 250, 'bag': 160, 'jewelry': 100, 'outerwear': 280},
        'luxury': {'top': 250, 'bottom': 350, 'dress': 500, 'shoes': 600, 'bag': 400, 'jewelry': 300, 'outerwear': 800}
    }

    # Item type mapping
    item_types = {
        'blazer': 'outerwear', 'cardigan': 'outerwear', 'jacket': 'outerwear',
        'blouse': 'top', 'sweater': 'top', 'top': 'top',
        'pants': 'bottom', 'jeans': 'bottom', 'trousers': 'bottom',
        'dress': 'dress',
        'shoes': 'shoes', 'pumps': 'shoes', 'heels': 'shoes', 'sneakers': 'shoes', 'boots': 'shoes',
        'bag': 'bag', 'clutch': 'bag',
        'jewelry': 'jewelry', 'necklace': 'jewelry', 'earrings': 'jewelry'
    }

    costs = cost_ranges.get(budget_category, cost_ranges['moderate'])
    total_cost = 0

    for item in items:
        # Determine item type
        item_type = 'top'  # default
        for key, value in item_types.items():
            if key in item.lower():
                item_type = value
                break

        total_cost += costs.get(item_type, costs['top'])

    return total_cost

def generate_shopping_categories(style_profile):
    """Generate shopping categories based on style profile"""

    lifestyle = style_profile.get('lifestyle_category', 'balanced')
    style_personality = style_profile.get('style_personality', 'eclectic')

    categories = {
        'essentials': ['basic_tees', 'jeans', 'blazer', 'little_black_dress', 'white_shirt'],
        'seasonal': ['summer_dresses', 'winter_coats', 'spring_jackets', 'fall_sweaters'],
        'accessories': ['bags', 'shoes', 'jewelry', 'scarves', 'belts'],
        'special_occasion': ['cocktail_dresses', 'formal_wear', 'evening_bags', 'dress_shoes']
    }

    # Customize based on lifestyle
    if lifestyle == 'professional':
        categories['work_wear'] = ['blazers', 'dress_pants', 'blouses', 'pencil_skirts', 'pumps']
    elif lifestyle == 'active':
        categories['activewear'] = ['workout_clothes', 'sneakers', 'athleisure', 'sports_bras']
    elif lifestyle == 'creative':
        categories['creative_pieces'] = ['statement_pieces', 'artistic_prints', 'unique_accessories']

    return categories

def generate_seasonal_recommendations(style_profile, preferences):
    """Generate seasonal outfit recommendations"""

    seasonal_pref = preferences.get('seasonal_preference', 'all_seasons')
    color_palette = style_profile.get('color_palette', {})

    seasons = {
        'spring': {
            'colors': ['pastels', 'light_colors', 'florals'],
            'items': ['light_jackets', 'midi_dresses', 'cardigans', 'flats'],
            'trends': ['floral_prints', 'pastel_colors', 'lightweight_fabrics']
        },
        'summer': {
            'colors': ['bright_colors', 'whites', 'light_blues'],
            'items': ['sundresses', 'shorts', 'sandals', 'tank_tops'],
            'trends': ['maxi_dresses', 'linen_fabrics', 'strappy_sandals']
        },
        'fall': {
            'colors': ['earth_tones', 'burgundy', 'mustard', 'brown'],
            'items': ['sweaters', 'boots', 'scarves', 'jackets'],
            'trends': ['layering', 'ankle_boots', 'cozy_sweaters']
        },
        'winter': {
            'colors': ['deep_colors', 'jewel_tones', 'black', 'navy'],
            'items': ['coats', 'boots', 'sweaters', 'scarves'],
            'trends': ['statement_coats', 'knee_high_boots', 'chunky_knits']
        }
    }

    return seasons

def generate_accessory_recommendations(style_profile, preferences, user=None):
    """Generate gender-specific accessory recommendations based on style profile"""

    style_personality = style_profile.get('style_personality', 'eclectic')
    accessory_prefs = preferences.get('accessory_preference', [])

    # Get user's gender for personalized recommendations
    user_gender = None
    if user and hasattr(user, 'gender'):
        user_gender = user.gender.lower() if user.gender else None

    # Default to 'unisex' if gender not specified
    if not user_gender or user_gender not in ['male', 'female']:
        user_gender = 'unisex'

    # Gender-specific accessories
    accessories = {
        'female': {
            'jewelry': {
                'classic': ['pearl_earrings', 'gold_chain', 'simple_rings', 'delicate_bracelet'],
                'trendy': ['statement_earrings', 'layered_necklaces', 'stackable_rings', 'charm_bracelet'],
                'bohemian': ['long_necklaces', 'chandelier_earrings', 'natural_stones', 'beaded_bracelet'],
                'minimalist': ['simple_studs', 'thin_chains', 'geometric_pieces', 'minimal_bracelet'],
                'edgy': ['bold_cuffs', 'statement_rings', 'dramatic_earrings', 'spike_bracelet']
            },
            'bags': {
                'classic': ['structured_tote', 'leather_crossbody', 'clutch', 'handbag'],
                'trendy': ['mini_bags', 'belt_bags', 'bucket_bags', 'saddle_bag'],
                'bohemian': ['fringe_bags', 'woven_totes', 'embroidered_clutches', 'hobo_bag'],
                'minimalist': ['simple_tote', 'clean_crossbody', 'structured_clutch', 'minimal_backpack'],
                'edgy': ['studded_bags', 'chain_bags', 'geometric_clutches', 'leather_backpack']
            },
            'shoes': {
                'classic': ['pumps', 'loafers', 'ballet_flats', 'ankle_boots'],
                'trendy': ['chunky_sneakers', 'platform_sandals', 'statement_heels', 'block_heels'],
                'bohemian': ['gladiator_sandals', 'fringe_boots', 'embellished_flats', 'wedge_sandals'],
                'minimalist': ['simple_flats', 'clean_sneakers', 'basic_heels', 'minimal_sandals'],
                'edgy': ['combat_boots', 'studded_heels', 'platform_boots', 'chunky_boots']
            },
            'accessories': {
                'classic': ['silk_scarf', 'leather_belt', 'sunglasses', 'hair_clips'],
                'trendy': ['bucket_hat', 'statement_belt', 'cat_eye_sunglasses', 'headband'],
                'bohemian': ['wide_brim_hat', 'braided_belt', 'round_sunglasses', 'hair_scarves'],
                'minimalist': ['simple_belt', 'minimal_sunglasses', 'hair_ties', 'basic_hat'],
                'edgy': ['studded_belt', 'dark_sunglasses', 'metal_hair_accessories', 'beanie']
            }
        },
        'male': {
            'jewelry': {
                'classic': ['simple_watch', 'wedding_ring', 'cufflinks', 'tie_clip'],
                'trendy': ['smart_watch', 'chain_necklace', 'signet_ring', 'bracelet'],
                'bohemian': ['leather_bracelet', 'beaded_necklace', 'wooden_ring', 'hemp_bracelet'],
                'minimalist': ['minimal_watch', 'simple_ring', 'thin_chain', 'basic_cufflinks'],
                'edgy': ['chunky_watch', 'skull_ring', 'thick_chain', 'metal_bracelet']
            },
            'bags': {
                'classic': ['leather_briefcase', 'messenger_bag', 'wallet', 'laptop_bag'],
                'trendy': ['crossbody_bag', 'backpack', 'sling_bag', 'tech_bag'],
                'bohemian': ['canvas_messenger', 'woven_bag', 'leather_satchel', 'vintage_briefcase'],
                'minimalist': ['simple_backpack', 'clean_messenger', 'minimal_wallet', 'basic_laptop_bag'],
                'edgy': ['leather_backpack', 'studded_messenger', 'chain_wallet', 'tactical_bag']
            },
            'shoes': {
                'classic': ['oxford_shoes', 'loafers', 'dress_boots', 'leather_sneakers'],
                'trendy': ['chunky_sneakers', 'high_top_sneakers', 'chelsea_boots', 'athletic_shoes'],
                'bohemian': ['suede_boots', 'canvas_sneakers', 'sandals', 'moccasins'],
                'minimalist': ['clean_sneakers', 'simple_boots', 'basic_loafers', 'minimal_dress_shoes'],
                'edgy': ['combat_boots', 'motorcycle_boots', 'platform_sneakers', 'studded_shoes']
            },
            'accessories': {
                'classic': ['leather_belt', 'sunglasses', 'tie', 'pocket_square'],
                'trendy': ['statement_belt', 'aviator_sunglasses', 'bow_tie', 'lapel_pin'],
                'bohemian': ['braided_belt', 'round_sunglasses', 'bandana', 'beaded_bracelet'],
                'minimalist': ['simple_belt', 'minimal_sunglasses', 'basic_tie', 'clean_watch'],
                'edgy': ['studded_belt', 'dark_sunglasses', 'leather_gloves', 'metal_accessories']
            }
        },
        'unisex': {
            'jewelry': {
                'classic': ['simple_watch', 'basic_ring', 'chain_necklace'],
                'trendy': ['smart_watch', 'stackable_rings', 'layered_chains'],
                'bohemian': ['leather_bracelet', 'natural_stones', 'beaded_jewelry'],
                'minimalist': ['minimal_watch', 'geometric_pieces', 'thin_chains'],
                'edgy': ['chunky_watch', 'statement_rings', 'bold_chains']
            },
            'bags': {
                'classic': ['backpack', 'crossbody_bag', 'tote_bag'],
                'trendy': ['sling_bag', 'belt_bag', 'bucket_bag'],
                'bohemian': ['canvas_bag', 'woven_tote', 'vintage_backpack'],
                'minimalist': ['simple_backpack', 'clean_tote', 'minimal_crossbody'],
                'edgy': ['leather_backpack', 'chain_bag', 'studded_bag']
            },
            'shoes': {
                'classic': ['sneakers', 'loafers', 'boots'],
                'trendy': ['chunky_sneakers', 'high_tops', 'platform_shoes'],
                'bohemian': ['canvas_shoes', 'suede_boots', 'sandals'],
                'minimalist': ['clean_sneakers', 'simple_boots', 'basic_shoes'],
                'edgy': ['combat_boots', 'platform_sneakers', 'studded_shoes']
            },
            'accessories': {
                'classic': ['sunglasses', 'belt', 'hat'],
                'trendy': ['statement_sunglasses', 'designer_belt', 'bucket_hat'],
                'bohemian': ['round_sunglasses', 'braided_belt', 'wide_brim_hat'],
                'minimalist': ['minimal_sunglasses', 'simple_belt', 'basic_hat'],
                'edgy': ['dark_sunglasses', 'studded_belt', 'beanie']
            }
        }
    }

    # Get gender-specific accessories
    gender_accessories = accessories.get(user_gender, accessories['unisex'])

    # Get recommendations for user's style personality
    recommendations = {}
    for category, styles in gender_accessories.items():
        recommendations[category] = styles.get(style_personality, styles['classic'])

    return recommendations

if __name__ == '__main__':
    with app.app_context():
        # Create all tables and add new columns if they don't exist
        db.create_all()

        # Check if new columns exist and add them if they don't
        try:
            # Try to access the new columns to see if they exist
            from sqlalchemy import text

            # Check for fashion_preferences column
            result = db.session.execute(text("PRAGMA table_info(user)")).fetchall()
            columns = [row[1] for row in result]

            if 'fashion_preferences' not in columns:
                print("Adding fashion_preferences column...")
                db.session.execute(text("ALTER TABLE user ADD COLUMN fashion_preferences TEXT"))

            if 'style_analysis_complete' not in columns:
                print("Adding style_analysis_complete column...")
                db.session.execute(text("ALTER TABLE user ADD COLUMN style_analysis_complete BOOLEAN DEFAULT 0"))

            db.session.commit()
            print("Database schema updated successfully!")

        except Exception as e:
            print(f"Database update error: {e}")
            db.session.rollback()

    app.run(debug=True)

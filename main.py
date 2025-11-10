from fastapi import FastAPI, Request, Form, Cookie, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import or_
import secrets
from typing import List, Dict

from database import SessionLocal, engine, Base
from models import User, Message, Post


'''
Helper unctions to calculate jackard_similarity for recommending users.
1. calculate_jaccard_similarity :- calculates jackard simmilarity between users
2. get_recommended_users :- Gets recommended users based on the similarity measure.
'''
def calculate_jaccard_similarity(interests1_str, interests2_str):
    """Calculates Jaccard Similarity between two comma-separated interest strings."""
    if not interests1_str or not interests2_str:
        return 0.0
    
    set1 = set(i.strip().lower() for i in interests1_str.split(',') if i.strip())
    set2 = set(i.strip().lower() for i in interests2_str.split(',') if i.strip())
    
    if not set1 or not set2:
        return 0.0
        
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def get_recommended_users(current_user, db: Session, limit=20):
    """Returns top 'limit' users based on interest similarity."""
    all_users = db.query(User).filter(User.id != current_user.id).all()
    
    user_scores = []
    for other_user in all_users:
        score = calculate_jaccard_similarity(current_user.interests, other_user.interests)
        # Only recommend if there is SOME similarity (score > 0)
        if score > 0:
            user_scores.append((other_user, score))
    
    # Sort by score descending, then take top 'limit'
    user_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [user for user, score in user_scores[:limit]]

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}

    async def connect(self, user_id: int, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_personal_message(self, message: str, user_id: int):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

manager = ConnectionManager()

# --- Database Setup ---
Base.metadata.create_all(bind=engine)

# --- FastAPI App Setup ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------- ROUTES -----------------

# --- Authentication & Landing Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Landing page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/check_email")
async def check_email(email: str = Form(...), db: Session = Depends(get_db)):
    """Checks if email exists to route to login or register."""
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return RedirectResponse(url=f"/login?email={email}", status_code=303)
    return RedirectResponse(url=f"/register?email={email}", status_code=303)

@app.get("/register", response_class=HTMLResponse)
async def register_form(request: Request, email: str = ""):
    """Registration page."""
    return templates.TemplateResponse("register.html", {"request": request, "email": email})

@app.post("/register")
async def register_user(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    department: str = Form(...),
    bio: str = Form(...),
    interests: List[str] = Form(default=[]),
    db: Session = Depends(get_db)
):
    """Handles new user registration."""
    interests_str = ", ".join(interests)
    token = secrets.token_urlsafe(16)

    user = User(
        name=name, email=email, password=password, department=department,
        bio=bio, interests=interests_str, session_token=token
    )
    db.add(user)
    db.commit()

    response = RedirectResponse(url="/home", status_code=303)
    response.set_cookie(key="session_token", value=token, httponly=True)
    return response

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, email: str = ""):
    """Login page."""
    return templates.TemplateResponse("login.html", {"request": request, "email": email})

@app.post("/login")
async def login_user(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    """Handles user login."""
    user = db.query(User).filter(User.email == email, User.password == password).first()
    if user:
        token = secrets.token_urlsafe(16)
        user.session_token = token
        db.commit()
        response = RedirectResponse(url="/home", status_code=303)
        response.set_cookie(key="session_token", value=token, httponly=True)
        return response
    return HTMLResponse("Invalid credentials. <a href='/'>Try again</a>")

@app.get("/logout")
async def logout(session_token: str = Cookie(None), db: Session = Depends(get_db)):
    """Logs out the user."""
    if session_token:
        user = db.query(User).filter(User.session_token == session_token).first()
        if user:
            user.session_token = None
            db.commit()
    response = RedirectResponse(url="/")
    response.delete_cookie("session_token")
    return response

#Added by Utkarsh for updating profiles and changing password.
@app.post("/update_profile")
async def update_profile(
    name: str = Form(...),
    bio: str = Form(...),
    session_token: str = Cookie(None),
    db: Session = Depends(get_db)
):
    if not session_token: return RedirectResponse(url="/")
    user = db.query(User).filter(User.session_token == session_token).first()
    if user:
        user.name = name
        user.bio = bio
        db.commit()
    # Redirect back to their own profile page
    return RedirectResponse(url=f"/profile/{user.id}", status_code=303)

#Simple password change (you can expand this with validation later)
@app.post("/change_password")
async def change_password(
    new_password: str = Form(...),
    session_token: str = Cookie(None),
    db: Session = Depends(get_db)
):
    if not session_token: return RedirectResponse(url="/")
    user = db.query(User).filter(User.session_token == session_token).first()
    if user:
        user.password = new_password
        db.commit()
    return RedirectResponse(url=f"/profile/{user.id}", status_code=303)

# --- Main Application Routes ---

@app.get("/home", response_class=HTMLResponse)
async def home(request: Request, session_token: str = Cookie(None), db: Session = Depends(get_db)):
    if not session_token: return RedirectResponse(url="/")
    user = db.query(User).filter(User.session_token == session_token).first()
    if not user: return RedirectResponse(url="/")

    # Fetch posts
    all_posts = db.query(Post).order_by(Post.timestamp.desc()).limit(50).all()
    
    #Fetch recommendations 
    recommendations = get_recommended_users(user, db, limit=15)

    return templates.TemplateResponse("home.html", {
        "request": request,
        "user": user,
        "posts": all_posts,
        "recommendations": recommendations 
    })

#Added by Utkarsh(to get users list)
@app.get("/users", response_class=HTMLResponse)
async def get_users_list(request: Request, session_token: str = Cookie(None), db: Session = Depends(get_db)):
    """Displays a list of all other users."""
    if not session_token: return RedirectResponse(url="/")
    current_user = db.query(User).filter(User.session_token == session_token).first()
    if not current_user: return RedirectResponse(url="/")

    all_users = db.query(User).filter(User.id != current_user.id).all()
    return templates.TemplateResponse("users.html", {"request": request, "users": all_users})

@app.get("/search", response_class=HTMLResponse)
async def search_users(
    request: Request,
    q: str = "",
    session_token: str = Cookie(None),
    db: Session = Depends(get_db)
):
    """Smart search: finds matches and sorts them by interest similarity."""
    if not session_token: return RedirectResponse(url="/")
    current_user = db.query(User).filter(User.session_token == session_token).first()
    if not current_user: return RedirectResponse(url="/")

    results = []
    if q.strip():
        # 1. Basic Search (Filter by name OR interest)
        matches = db.query(User).filter(
            User.id != current_user.id,
            or_(
                User.name.ilike(f"%{q}%"),
                User.interests.ilike(f"%{q}%")
            )
        ).all()

        # 2. Smart Sort (Calculate similarity for matches only) - O(N)
        scored_results = []
        for user in matches:
            score = calculate_jaccard_similarity(current_user.interests, user.interests)
            scored_results.append((user, score))
        
        # 3. Sort by score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Unpack just the user objects for the template
        results = [user for user, score in scored_results]

    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "user": current_user,
            "query": q,
            "results": results
        }
    )

@app.get("/profile", response_class=HTMLResponse)
async def profile(request: Request, session_token: str = Cookie(None), db: Session = Depends(get_db)):
    """Dedicated profile page (optional, if used)."""
    if not session_token: return RedirectResponse(url="/")
    user = db.query(User).filter(User.session_token == session_token).first()
    if not user: return RedirectResponse(url="/")

    my_posts = db.query(Post).filter(Post.user_id == user.id).order_by(Post.timestamp.desc()).all()
    return templates.TemplateResponse("profile.html", {"request": request, "user": user, "posts": my_posts})

#Added by Utkarsh to view user profile
@app.get("/profile/{user_id}", response_class=HTMLResponse)
async def view_profile(
    request: Request,
    user_id: int,
    session_token: str = Cookie(None),
    db: Session = Depends(get_db)
):
    """View any user's profile."""
    if not session_token: return RedirectResponse(url="/")
    current_user = db.query(User).filter(User.session_token == session_token).first()
    if not current_user: return RedirectResponse(url="/")

    # Fetch the user whose profile we want to see
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        return HTMLResponse("User not found", status_code=404)

    # Fetch their posts only
    user_posts = db.query(Post).filter(Post.user_id == user_id).order_by(Post.timestamp.desc()).all()

    return templates.TemplateResponse("user_profile.html", {
        "request": request,
        "user": current_user,     # The person looking (for navbar)
        "target_user": target_user, # The profile being looked at
        "posts": user_posts
    })

# --- Feature Routes (Posts, Chat, Interests) ---

@app.post("/create_post")
async def create_post(content: str = Form(...), session_token: str = Cookie(None), db: Session = Depends(get_db)):
    """Creates a new post."""
    if not session_token: return RedirectResponse(url="/", status_code=303)
    user = db.query(User).filter(User.session_token == session_token).first()
    if user:
        new_post = Post(user_id=user.id, content=content)
        db.add(new_post)
        db.commit()
    return RedirectResponse(url="/home", status_code=303)

@app.post("/update_interests")
async def update_interests(interests: dict = {}, session_token: str = Cookie(None), db: Session = Depends(get_db)):
    """Updates user interests via JSON."""
    if not session_token: return JSONResponse({"error": "Not authenticated"}, status_code=401)
    user = db.query(User).filter(User.session_token == session_token).first()
    if user:
        user.interests = ", ".join(interests.get("interests", []))
        db.commit()
    return JSONResponse({"status": "success"})

#Added by Utkarsh (Route for chat)
@app.get("/chat/{receiver_id}", response_class=HTMLResponse)
async def get_chat_page(request: Request, receiver_id: int, session_token: str = Cookie(None), db: Session = Depends(get_db)):
    """Serves the chat page with message history."""
    if not session_token: return RedirectResponse(url="/")
    current_user = db.query(User).filter(User.session_token == session_token).first()
    if not current_user: return RedirectResponse(url="/")

    receiver_user = db.query(User).filter(User.id == receiver_id).first()
    if not receiver_user: return HTMLResponse("User not found.", status_code=404)

    message_history = db.query(Message).filter(
        or_(
            (Message.sender_id == current_user.id) & (Message.receiver_id == receiver_id),
            (Message.sender_id == receiver_id) & (Message.receiver_id == current_user.id)
        )
    ).order_by(Message.timestamp.asc()).all()

    return templates.TemplateResponse("chat.html", {
        "request": request, "current_user": current_user, "receiver": receiver_user, "message_history": message_history
    })

#Added by Utkarsh(websocket for chatting feature)
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int, db: Session = Depends(get_db)):
    """Handles real-time chat via WebSockets."""
    await manager.connect(user_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            sender_id = data['sender_id']
            receiver_id = data['receiver_id']
            message_content = data['content']
            
            db_message = Message(sender_id=sender_id, receiver_id=receiver_id, content=message_content)
            db.add(db_message)
            db.commit()

            await manager.send_personal_message(message_content, receiver_id)
    except WebSocketDisconnect:
        manager.disconnect(user_id)
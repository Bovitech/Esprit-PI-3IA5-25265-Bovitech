from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from .models import Farm

def generate_tokens(user):
    refresh = RefreshToken.for_user(user)
    return {
        'token': str(refresh.access_token),
        'refresh': str(refresh),
    }

def register_user(data):
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    farm_name = data.get('farm_name')
    cow_count = data.get('cow_count')
    region = data.get('region')

    if User.objects.filter(username=username).exists():
        return None, 'Username déjà utilisé'
    if User.objects.filter(email=email).exists():
        return None, 'Email déjà utilisé'

    user = User.objects.create_user(username=username, email=email, password=password)
    
    Farm.objects.create(
        user=user,
        farm_name=farm_name or '',
        cow_count=cow_count or 0,
        region=region or '',
    )
    
    return generate_tokens(user), None

def login_user(email, password):
    try:
        print(email, password)
        user = User.objects.get(email=email)
        user = authenticate(username=user.username, password=password)
        if not user:
            return None, 'Identifiants incorrects'
        return generate_tokens(user), None
    except User.DoesNotExist:
        return None, 'Identifiants incorrects'

def logout_user(refresh_token):
    try:
        token = RefreshToken(refresh_token)
        token.blacklist()
        return True, None
    except Exception:
        return False, 'Token invalide'
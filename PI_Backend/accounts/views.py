from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from .services import register_user, login_user, logout_user
from .utils import success_response, error_response
from .models import Farm

class RegisterView(APIView):
    def post(self, request):
        tokens, error = register_user(request.data)
        if error:
            return error_response(error)
        return success_response(tokens)

class LoginView(APIView):
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        tokens, error = login_user(email, password)
        if error:
            return error_response(error, 401)
        return success_response(tokens)

class LogoutView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        success, error = logout_user(request.data.get('refresh'))
        if error:
            return error_response(error)
        return success_response({'message': 'Déconnexion réussie'})

class ProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        try:
            farm = Farm.objects.get(user=user)
            farm_data = {
                'farm_name': farm.farm_name,
                'cow_count': farm.cow_count,
                'region': farm.region,
            }
        except Farm.DoesNotExist:
            farm_data = None

        return success_response({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'farm': farm_data,
        })
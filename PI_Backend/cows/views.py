from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .models import Cow
from .serializers import CowSerializer
from accounts.models import Farm


class CowListCreateView(APIView):
    permission_classes = [IsAuthenticated]

    # GET all cows of logged user's farm
    def get(self, request):
        cows = Cow.objects.filter(farm__user=request.user)
        serializer = CowSerializer(cows, many=True)
        return Response({"data": serializer.data})

    # CREATE cow
    def post(self, request):
        farm = Farm.objects.get(user=request.user)

        data = request.data.copy()
        data['farm'] = farm.id

        serializer = CowSerializer(data=data)

        if serializer.is_valid():
            serializer.save()
            return Response({"data": serializer.data})

        return Response(serializer.errors, status=400)
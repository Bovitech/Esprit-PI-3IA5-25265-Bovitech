from rest_framework.response import Response
from rest_framework import status

def success_response(data, status_code=status.HTTP_200_OK):
    return Response({'success': True, 'data': data}, status=status_code)

def error_response(message, status_code=status.HTTP_400_BAD_REQUEST):
    return Response({'success': False, 'error': message}, status=status_code)
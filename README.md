#  Login Inteligente

Sistema de autenticación facial seguro con detección de vida (liveness detection) desarrollado en Python.

##  Características

- Registro con verificación por correo electrónico
- Detección de parpadeo real (anti-spoofing)
- Reconocimiento facial con DeepFace
- Código alternativo por correo si falla el reconocimiento

##  Requisitos

pip install opencv-python mediapipe customtkinter mtcnn deepface tf-keras python-dotenv

##  Configuración

1. Clona el repositorio
2. Copia `.env.example` y renómbralo a `.env`
3. Llena tus credenciales en `.env`:
EMAIL_REMITENTE=tu_correo@gmail.com
APP_PASSWORD=tu_app_password_de_gmail

4. Ejecuta el script:
python Login_Vision.py


##  Notas

- Requiere Python 3.11
- La primera vez descarga automáticamente el modelo de MediaPipe (~3 MB)
- Para obtener tu App Password de Gmail: Cuenta de Google → Seguridad → Verificación en dos pasos → Contraseñas de aplicación
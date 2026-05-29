# ─────────────────────────────────────────────────────────────
# LIBRERÍAS
# ─────────────────────────────────────────────────────────────
import os
import cv2
import numpy as np
import smtplib
import random
import urllib.request
import mediapipe as mp
import customtkinter as ctk
from tkinter import messagebox
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE CORREO
# ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
EMAIL_REMITENTE = os.getenv("EMAIL_REMITENTE")
APP_PASSWORD    = os.getenv("APP_PASSWORD")

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE INTERFAZ
# ─────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────
def porcentaje_oscuridad(imagen):
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binarizada = cv2.threshold(imagen_gris, 50, 255, cv2.THRESH_BINARY)
    total = imagen.shape[0] * imagen.shape[1]
    oscuros = np.count_nonzero(binarizada == 0)
    return (oscuros / total) * 100

def mostrar_destello(duration=1000):
    flash = np.ones((1080, 1920), np.uint8) * 255
    cv2.imshow('flash', flash)
    cv2.moveWindow('flash', 0, 0)
    cv2.waitKey(duration)

def generar_codigo():
    return str(random.randint(100000, 999999))

def enviar_codigo(correo_destino, nombre_usuario, tipo="registro"):
    codigo = generar_codigo()
    if tipo == "registro":
        asunto = " Confirma tu registro — Login Inteligente"
        accion = "completar tu registro"
    else:
        asunto = " Código de acceso — Login Inteligente"
        accion = "acceder a tu cuenta"

    cuerpo_html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #f4f4f4;
                 padding: 30px; margin: 0;">
        <div style="max-width: 420px; margin: auto; background-color: white;
                    border-radius: 12px; padding: 30px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <h2 style="color: #1a1a1a; text-align: center; margin-bottom: 5px;">
                 Login Inteligente
            </h2>
            <p style="color: #666; text-align: center; margin-top: 0;">
                Hola <strong>{nombre_usuario}</strong>, usa este código para {accion}:
            </p>
            <div style="background-color: #f0f6ff; border: 2px solid #4A9EFF;
                        border-radius: 12px; padding: 25px;
                        text-align: center; margin: 20px 0;">
                <p style="font-size: 42px; font-weight: bold;
                           color: #4A9EFF; letter-spacing: 10px;
                           margin: 0; font-family: monospace;">
                    {codigo}
                </p>
            </div>
            <p style="color: #999; text-align: center; font-size: 13px;">
                 Este código expira en <strong>5 minutos</strong>.
            </p>
            <p style="color: #999; text-align: center; font-size: 12px;
                      border-top: 1px solid #eee; padding-top: 15px;">
                 No compartas este código con nadie.
            </p>
        </div>
    </body>
    </html>
    """
    try:
        msg = MIMEMultipart('alternative')
        msg['From']    = EMAIL_REMITENTE
        msg['To']      = correo_destino
        msg['Subject'] = asunto
        msg.attach(MIMEText(cuerpo_html, 'html'))
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(EMAIL_REMITENTE, APP_PASSWORD)
        servidor.sendmail(EMAIL_REMITENTE, correo_destino, msg.as_string())
        servidor.quit()
        return codigo
    except Exception as e:
        print(f"Error al enviar correo: {e}")
        return None

def leer_datos_usuario(nombre):
    try:
        with open(nombre, "r") as f:
            lineas = f.read().splitlines()
        return lineas
    except:
        return None

# ─────────────────────────────────────────────────────────────
# LIVENESS DETECTION CON MEDIAPIPE 0.10.x (EAR)
# ─────────────────────────────────────────────────────────────
def detectar_parpadeo():
    """Detecta parpadeo real usando EAR con MediaPipe 0.10.x. Retorna True si detecta 2 parpadeos."""

    # Descargar el modelo si no existe
    modelo_path = "face_landmarker.task"
    if not os.path.exists(modelo_path):
        print("Descargando modelo face_landmarker.task (~3 MB)...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        try:
            urllib.request.urlretrieve(url, modelo_path)
            print("Modelo descargado correctamente.")
        except Exception as e:
            print(f"Error descargando modelo: {e}")
            return False

    # Configurar la nueva API de MediaPipe Tasks
    BaseOptions        = mp.tasks.BaseOptions
    FaceLandmarker     = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode  = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=modelo_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )

    # Índices de landmarks de ojos en MediaPipe Face Mesh
    OJO_IZQ = [362, 385, 387, 263, 373, 380]
    OJO_DER = [33,  160, 158, 133, 153, 144]

    def calcular_ear(landmarks, indices, w, h):
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        if C == 0:
            return 0
        return (A + B) / (2.0 * C)

    EAR_UMBRAL   = 0.22
    parpadeos    = 0
    ojo_cerrado  = False
    intentos     = 0
    MAX_INTENTOS = 400

    cap = cv2.VideoCapture(0)

    with FaceLandmarker.create_from_options(options) as landmarker:
        while intentos < MAX_INTENTOS:
            ret, frame = cap.read()
            if not ret:
                break
            intentos += 1

            h, w      = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Crear imagen MediaPipe y procesar
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            resultado = landmarker.detect(mp_image)

            cv2.putText(frame, f"Parpadea naturalmente | Parpadeos: {parpadeos}/2",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(frame, "ESC para cancelar",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            if resultado.face_landmarks:
                lm      = resultado.face_landmarks[0]   # lista de landmarks
                ear_izq = calcular_ear(lm, OJO_IZQ, w, h)
                ear_der = calcular_ear(lm, OJO_DER, w, h)
                ear_avg = (ear_izq + ear_der) / 2.0

                cv2.putText(frame, f"EAR: {ear_avg:.2f}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                if ear_avg < EAR_UMBRAL and not ojo_cerrado:
                    ojo_cerrado = True
                elif ear_avg >= EAR_UMBRAL and ojo_cerrado:
                    ojo_cerrado = False
                    parpadeos  += 1
                    print(f"Parpadeo detectado #{parpadeos}")
            else:
                cv2.putText(frame, "Acercate mas a la camara",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

            cv2.imshow("Liveness — Parpadea frente a la camara", frame)

            if parpadeos >= 2:
                break

            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    return parpadeos >= 2

# ─────────────────────────────────────────────────────────────
# CAPTURA FACIAL
# ─────────────────────────────────────────────────────────────
def capturar_rostro(nombre_archivo, titulo_ventana):
    cap = cv2.VideoCapture(0)
    frame_guardado = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = face_cascade.detectMultiScale(
            frame, scaleFactor=1.1, minNeighbors=5
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        porcentaje = porcentaje_oscuridad(frame)
        cv2.putText(
            frame,
            f'Oscuridad: {porcentaje:.1f}% | ESC para capturar',
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        if porcentaje > 50:
            mostrar_destello(2000)
        cv2.imshow(titulo_ventana, frame)
        frame_guardado = frame
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    if frame_guardado is not None:
        cv2.imwrite(nombre_archivo + ".jpg", frame_guardado)
        try:
            pixeles  = pyplot.imread(nombre_archivo + ".jpg")
            detector = MTCNN()
            caras    = detector.detect_faces(pixeles)
            if caras:
                x1, y1, ancho, alto = caras[0]['box']
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = x1 + ancho, y1 + alto
                cara     = pixeles[y1:y2, x1:x2]
                cara     = cv2.resize(cara, (150, 200), interpolation=cv2.INTER_CUBIC)
                cara_bgr = cv2.cvtColor(cara, cv2.COLOR_RGB2BGR)
                cv2.imwrite(nombre_archivo + ".jpg", cara_bgr)
                return True
        except Exception as e:
            print(f"Error MTCNN: {e}")
    return False

# ─────────────────────────────────────────────────────────────
# COMPARACIÓN FACIAL CON DEEPFACE
# ─────────────────────────────────────────────────────────────
def comparar_rostros(img_registro, img_login):
    try:
        cv2.imwrite("temp_reg.jpg", img_registro)
        cv2.imwrite("temp_log.jpg", img_login)

        from deepface import DeepFace
        resultado = DeepFace.verify(
            img1_path="temp_reg.jpg",
            img2_path="temp_log.jpg",
            model_name="VGG-Face",
            enforce_detection=False
        )
        distancia = resultado["distance"]
        similitud = 1 - distancia
        similitud = max(0.0, min(1.0, similitud))
        print(f"DeepFace distancia: {distancia:.4f} | similitud: {similitud:.4f}")
        return similitud

    except Exception as e:
        print(f"Error deepface: {e}")
        return 0

# ─────────────────────────────────────────────────────────────
# VENTANA DE CÓDIGO
# ─────────────────────────────────────────────────────────────
def ventana_verificar_codigo(padre, codigo_correcto, callback_exito, subtitulo=""):
    win = ctk.CTkToplevel(padre)
    win.title("Verificación de código")
    win.geometry("380x220")
    win.resizable(False, False)
    win.grab_set()

    ctk.CTkLabel(
        win, text="📧 Código enviado a tu correo",
        font=ctk.CTkFont(size=16, weight="bold")
    ).pack(pady=15)

    if subtitulo:
        ctk.CTkLabel(win, text=subtitulo,
                     font=ctk.CTkFont(size=12),
                     text_color="gray").pack()

    codigo_var = ctk.StringVar()
    ctk.CTkEntry(
        win, textvariable=codigo_var,
        width=220, height=42,
        placeholder_text="Ingresa el código de 6 dígitos",
        font=ctk.CTkFont(size=15)
    ).pack(pady=12)

    def verificar():
        if codigo_var.get().strip() == codigo_correcto:
            win.destroy()
            callback_exito()
        else:
            messagebox.showerror(" X Código incorrecto",
                                 "El código no coincide. Intenta de nuevo.")

    ctk.CTkButton(
        win, text="Verificar", width=220, height=40,
        command=verificar
    ).pack()

# ─────────────────────────────────────────────────────────────
# CÓDIGO ALTERNATIVO
# ─────────────────────────────────────────────────────────────
def ofrecer_codigo_alternativo(nombre, correo):
    respuesta = messagebox.askyesno(
        " X Verificación facial fallida",
        "El rostro no coincide con el registrado.\n\n"
        "¿Deseas recibir un código de acceso\n"
        "a tu correo como método alternativo?"
    )
    if respuesta and correo:
        codigo = enviar_codigo(correo, nombre, tipo="login")
        if codigo:
            def acceso_por_codigo():
                messagebox.showinfo(
                    " ✔ ",
                    " Bienvenido",
                    "Acceso concedido por código de correo."
                )
                pantalla2.destroy()
            ventana_verificar_codigo(
                pantalla2, codigo, acceso_por_codigo,
                subtitulo=f"Código enviado a {correo}"
            )
        else:
            messagebox.showerror(" X Error",
                                 "No se pudo enviar el código.")
    elif respuesta and not correo:
        messagebox.showerror(" X Sin correo",
                             "No hay correo registrado.")

# ─────────────────────────────────────────────────────────────
# REGISTRO
# ─────────────────────────────────────────────────────────────
def registro():
    global pantalla1

    pantalla1 = ctk.CTkToplevel(pantalla)
    pantalla1.title("Registro de Usuario")
    pantalla1.geometry("400x500")
    pantalla1.resizable(False, False)

    ctk.CTkLabel(
        pantalla1, text="📋 Nuevo Registro",
        font=ctk.CTkFont(size=20, weight="bold")
    ).pack(pady=20)

    ctk.CTkLabel(
        pantalla1,
        text="Completa los 3 campos y captura tu rostro",
        font=ctk.CTkFont(size=12), text_color="gray"
    ).pack()

    usuario_var = ctk.StringVar()
    contra_var  = ctk.StringVar()
    correo_var  = ctk.StringVar()

    ctk.CTkLabel(pantalla1, text="Usuario *",
                 font=ctk.CTkFont(size=13)).pack(anchor="w", padx=50, pady=(15,0))
    ctk.CTkEntry(
        pantalla1, textvariable=usuario_var,
        width=300, height=38,
        placeholder_text="Nombre de usuario"
    ).pack()

    ctk.CTkLabel(pantalla1, text="Contraseña *",
                 font=ctk.CTkFont(size=13)).pack(anchor="w", padx=50, pady=(10,0))
    ctk.CTkEntry(
        pantalla1, textvariable=contra_var,
        width=300, height=38, show="*",
        placeholder_text="Contraseña"
    ).pack()

    ctk.CTkLabel(pantalla1, text="Correo electrónico *",
                 font=ctk.CTkFont(size=13)).pack(anchor="w", padx=50, pady=(10,0))
    ctk.CTkEntry(
        pantalla1, textvariable=correo_var,
        width=300, height=38,
        placeholder_text="correo@ejemplo.com"
    ).pack()

    lbl_estado = ctk.CTkLabel(pantalla1, text="",
                               font=ctk.CTkFont(size=12))

    def iniciar_registro():
        nombre = usuario_var.get().strip()
        contra = contra_var.get().strip()
        correo = correo_var.get().strip()

        if not nombre or not contra or not correo:
            messagebox.showwarning("⚠️ Campos vacíos",
                                   "Debes completar usuario, contraseña y correo.")
            return
        if "@" not in correo or "." not in correo:
            messagebox.showerror(" X Correo inválido",
                                 "Ingresa un correo electrónico válido.")
            return
        if os.path.exists(nombre):
            messagebox.showerror(" X Usuario existente",
                                 "Ese nombre de usuario ya está registrado.")
            return

        lbl_estado.configure(text="Enviando código al correo...", text_color="orange")
        pantalla1.update()
        codigo = enviar_codigo(correo, nombre, tipo="registro")

        if not codigo:
            messagebox.showerror(" X Error de correo",
                                 "No se pudo enviar el código. Verifica el correo.")
            lbl_estado.configure(text="", text_color="gray")
            return

        lbl_estado.configure(text="Código enviado ✓", text_color="green")

        def continuar_con_facial():
            with open(nombre, "w") as f:
                f.write(nombre + "\n")
                f.write(contra + "\n")
                f.write(correo)
            lbl_estado.configure(text="Abriendo cámara...", text_color="orange")
            pantalla1.update()
            exito = capturar_rostro(nombre, "Registro Facial — ESC para capturar")
            if exito:
                lbl_estado.configure(text="✔ Registro completo", text_color="green")
                messagebox.showinfo(" ✔ Registro exitoso",
                                    f"Bienvenido {nombre}!\nYa puedes iniciar sesión.")
                pantalla1.destroy()
            else:
                lbl_estado.configure(text=" X No se detectó rostro", text_color="red")
                messagebox.showerror(" X Error facial",
                                     "No se detectó un rostro. Intenta de nuevo.")

        ventana_verificar_codigo(
            pantalla1, codigo, continuar_con_facial,
            subtitulo=f"Código enviado a {correo}"
        )

    ctk.CTkButton(
        pantalla1, text="📝 Registrarme",
        width=300, height=42,
        font=ctk.CTkFont(size=14),
        command=iniciar_registro
    ).pack(pady=20)

    lbl_estado.pack()

# ─────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────
def login():
    global pantalla2

    pantalla2 = ctk.CTkToplevel(pantalla)
    pantalla2.title("Iniciar Sesión")
    pantalla2.geometry("400x420")
    pantalla2.resizable(False, False)

    ctk.CTkLabel(
        pantalla2, text="🔐 Iniciar Sesión",
        font=ctk.CTkFont(size=20, weight="bold")
    ).pack(pady=20)

    ctk.CTkLabel(
        pantalla2,
        text="Paso 1: ingresa tus credenciales",
        font=ctk.CTkFont(size=12), text_color="gray"
    ).pack()

    usuario_var = ctk.StringVar()
    contra_var  = ctk.StringVar()

    ctk.CTkLabel(pantalla2, text="Usuario *",
                 font=ctk.CTkFont(size=13)).pack(anchor="w", padx=50, pady=(15,0))
    ctk.CTkEntry(
        pantalla2, textvariable=usuario_var,
        width=300, height=38,
        placeholder_text="Nombre de usuario"
    ).pack()

    ctk.CTkLabel(pantalla2, text="Contraseña *",
                 font=ctk.CTkFont(size=13)).pack(anchor="w", padx=50, pady=(10,0))
    ctk.CTkEntry(
        pantalla2, textvariable=contra_var,
        width=300, height=38, show="*",
        placeholder_text="Contraseña"
    ).pack()

    lbl_estado = ctk.CTkLabel(pantalla2, text="",
                               font=ctk.CTkFont(size=12))

    def verificar_credenciales():
        nombre = usuario_var.get().strip()
        contra = contra_var.get().strip()

        if not nombre or not contra:
            messagebox.showwarning("⚠ Campos vacíos",
                                   "Ingresa usuario y contraseña.")
            return

        datos = leer_datos_usuario(nombre)
        if not datos:
            messagebox.showerror("X Usuario no encontrado",
                                 "No existe ese usuario. ¿Ya te registraste?")
            return

        if contra != datos[1]:
            messagebox.showerror("X Contraseña incorrecta",
                                 "La contraseña no es correcta.")
            return

        lbl_estado.configure(
            text="✓ Credenciales correctas — Paso 2: liveness",
            text_color="green"
        )
        pantalla2.update()
        correo_usuario = datos[2] if len(datos) > 2 else None
        iniciar_verificacion_facial(nombre, correo_usuario)

    def iniciar_verificacion_facial(nombre, correo):

        # ── PASO 2: LIVENESS ─────────────────────────────────
        lbl_estado.configure(text="Verificando que eres real...", text_color="orange")
        pantalla2.update()

        es_real = detectar_parpadeo()
        if not es_real:
            lbl_estado.configure(text="X Liveness fallido", text_color="red")
            messagebox.showerror(
                " X No se detectó vida",
                "No se detectaron 2 parpadeos.\n"
                "Mira de frente a la cámara y parpadea\n"
                "de forma natural 2 veces.\n\n"
                "Esto evita el uso de fotos falsas."
            )
            ofrecer_codigo_alternativo(nombre, correo)
            return

        # ── PASO 3: CAPTURA FACIAL ────────────────────────────
        lbl_estado.configure(text="Abriendo cámara...", text_color="orange")
        pantalla2.update()

        cap = cv2.VideoCapture(0)
        frame_login = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = face_cascade.detectMultiScale(
                frame, scaleFactor=1.1, minNeighbors=5
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            porcentaje = porcentaje_oscuridad(frame)
            cv2.putText(
                frame,
                f'Oscuridad: {porcentaje:.1f}% | ESC para verificar',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            if porcentaje > 50:
                mostrar_destello(2000)
            cv2.imshow("Login Facial — ESC para verificar", frame)
            frame_login = frame
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

        # ── VALIDACIÓN 1: Haar detecta rostro ────────────────
        faces_check = face_cascade.detectMultiScale(
            frame_login, scaleFactor=1.1, minNeighbors=5
        )
        if len(faces_check) == 0:
            lbl_estado.configure(text="X No se detectó rostro", text_color="red")
            messagebox.showerror(
                "X Sin rostro detectado",
                "No se detectó ningún rostro.\n"
                "Asegúrate de estar frente a la cámara\n"
                "con buena iluminación."
            )
            ofrecer_codigo_alternativo(nombre, correo)
            return

        # Guardar foto
        archivo_login = nombre + "LOG"
        cv2.imwrite(archivo_login + ".jpg", frame_login)

        # ── VALIDACIÓN 2: MTCNN recorta rostro ───────────────
        try:
            pixeles  = pyplot.imread(archivo_login + ".jpg")
            detector = MTCNN()
            caras    = detector.detect_faces(pixeles)

            if not caras:
                lbl_estado.configure(text="X Rostro no procesado", text_color="red")
                messagebox.showerror(
                    "X Rostro no válido",
                    "No se pudo procesar el rostro.\n"
                    "Intenta con mejor iluminación y de frente."
                )
                ofrecer_codigo_alternativo(nombre, correo)
                return

            x1, y1, ancho, alto = caras[0]['box']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = x1 + ancho, y1 + alto
            cara = pixeles[y1:y2, x1:x2]

            if cara.size == 0:
                messagebox.showerror("❌ Error", "No se pudo recortar el rostro.")
                ofrecer_codigo_alternativo(nombre, correo)
                return

            cara     = cv2.resize(cara, (150, 200), interpolation=cv2.INTER_CUBIC)
            cara_bgr = cv2.cvtColor(cara, cv2.COLOR_RGB2BGR)
            cv2.imwrite(archivo_login + ".jpg", cara_bgr)

        except Exception as e:
            print(f"Error MTCNN login: {e}")
            ofrecer_codigo_alternativo(nombre, correo)
            return

        # ── VALIDACIÓN 3: existe foto de registro ─────────────
        if not os.path.exists(nombre + ".jpg"):
            messagebox.showerror(
                "❌ Sin foto de registro",
                "No se encontró la foto del registro.\n"
                "Por favor regístrate nuevamente."
            )
            return

        # ── COMPARACIÓN DEEPFACE ──────────────────────────────
        rostro_reg = cv2.imread(nombre + ".jpg")
        rostro_log = cv2.imread(archivo_login + ".jpg")

        if rostro_reg is None or rostro_log is None:
            messagebox.showerror("X Error", "No se pudieron cargar las imágenes.")
            return

        rostro_reg = cv2.resize(rostro_reg, (150, 200))
        rostro_log = cv2.resize(rostro_log, (150, 200))

        similitud = comparar_rostros(rostro_reg, rostro_log)
        print(f"Similitud facial final: {similitud:.4f} ({similitud*100:.1f}%)")

        if similitud >= 0.50:
            lbl_estado.configure(text="✔ Acceso concedido", text_color="green")
            messagebox.showinfo(
                "✔ Bienvenido",
                f"Hola {nombre}!\n"
                f"Similitud facial: {similitud*100:.1f}%\n\n"
                "Acceso concedido."
            )
            pantalla2.destroy()
        else:
            lbl_estado.configure(
                text=f"X Similitud {similitud*100:.1f}% — mínimo 50%",
                text_color="red"
            )
            ofrecer_codigo_alternativo(nombre, correo)

    ctk.CTkButton(
        pantalla2, text="Ingresar →",
        width=300, height=42,
        font=ctk.CTkFont(size=14),
        command=verificar_credenciales
    ).pack(pady=20)

    lbl_estado.pack()

# ─────────────────────────────────────────────────────────────
# PANTALLA PRINCIPAL
# ─────────────────────────────────────────────────────────────
def pantalla_principal():
    global pantalla
    pantalla = ctk.CTk()
    pantalla.geometry("400x350")
    pantalla.title("Login Inteligente")
    pantalla.resizable(False, False)

    ctk.CTkLabel(
        pantalla, text="🔐 Login Inteligente",
        font=ctk.CTkFont(size=24, weight="bold")
    ).pack(pady=30)

    ctk.CTkLabel(
        pantalla,
        text="Sistema de autenticación facial seguro",
        font=ctk.CTkFont(size=13), text_color="gray"
    ).pack()

    ctk.CTkButton(
        pantalla, text="Iniciar Sesión",
        width=250, height=48,
        corner_radius=10,
        font=ctk.CTkFont(size=15),
        command=login
    ).pack(pady=25)

    ctk.CTkButton(
        pantalla, text="Registrarse",
        width=250, height=48,
        corner_radius=10,
        font=ctk.CTkFont(size=15),
        fg_color="transparent",
        border_width=2,
        command=registro
    ).pack()

    ctk.CTkLabel(
        pantalla,
        text="Registro requiere: usuario + contraseña + correo + foto facial",
        font=ctk.CTkFont(size=10), text_color="gray"
    ).pack(pady=20)

    pantalla.mainloop()

pantalla_principal()

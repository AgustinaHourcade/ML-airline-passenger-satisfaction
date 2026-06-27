# Usa la imagen oficial de Python ligera
FROM python:3.12-slim

# Establecer directorio de trabajo
WORKDIR /app

# Crear un usuario no-root (requerido por las políticas de seguridad de Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

# Ajustar el PATH para que pip instale binarios del usuario
ENV PATH="/home/user/.local/bin:${PATH}"

# Copiar el archivo de requirements e instalar dependencias
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código fuente al contenedor
COPY --chown=user . .

# Exponer el puerto 7860 (puerto por defecto en Hugging Face Spaces)
EXPOSE 7860

# Comando para iniciar la API con Uvicorn en el puerto 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]

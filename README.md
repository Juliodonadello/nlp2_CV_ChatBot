# 🚀 Asistente de CV con RAG 100% Gratuito

**Chatbot especializado en análisis de currículums** usando tecnología completamente gratuita.  
Utiliza el plan free tier de Pinecone, embeddings de código abierto, y el acceso gratuito a ChatGPT del GitHub Marketplace.

## 🌟 Características Clave
- **Costo Cero**: Sin APIs pagas
- **Memoria Conversacional**: Mantiene contexto del diálogo (últimas 5 interacciones)
- **Tecnología Gratuita**:
  - 🤗 Embeddings con `all-MPNet-base-v2` (768 dimensiones)
  - 🗄️ Vector DB en Pinecone (Free Tier)
  - 🤖 Modelo GPT-4 via GitHub Student Developer Pack (Marketplace)

## 🛠️ Componentes Técnicos
| Componente               | Tecnología                     | Especificaciones               |
|--------------------------|--------------------------------|---------------------------------|
| Procesamiento de Texto    | PyPDF2                         | Extracción de PDFs             |
| Embeddings                | Hugging Face Transformers      | Modelo `all-MPNet-base-v2` (768D) |
| Vector Database           | Pinecone                       | Índice de 768 dimensiones       |
| Modelo de Chat            | GPT-4 (via Azure)              | GitHub Marketplace Free Tier   |
| Framework                 | LangChain + Streamlit          | Interfaz web                   |

## 📋 Requisitos Previos
1. Cuentas gratuitas:
   - [GitHub Education](https://education.github.com/pack) (para API Key de Azure)
   - [Pinecone](https://www.pinecone.io/) (crear índice de 768D)
   - [Azure for Students](https://azure.microsoft.com/es-es/free/students/)

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuración

Crear archivo `API_KEYS.txt` en la raíz del proyecto con:

```txt
PINECONE_API_KEY=tu_key_pinecone
GITHUB_TOKEN=tu_token_github_education
PINECONE_INDEX_NAME=nombre_indice_768d
```

En Pinecone:

- Crear índice con **768 dimensiones**
- Usar métrica **cosine**
- Seleccionar región **gcp-starter**

---

## 🔄 Flujo de Trabajo

1. **Carga de PDF** → `Curriculum.pdf`
2. **Procesamiento**:
   - Limpieza de texto
   - Chunking (500 tokens)
3. **Embeddings**:
   - Transformers (768D)
4. **Almacenamiento**:
   - Pinecone
5. **Consulta**:
   - Búsqueda semántica (MMR)
   - Contexto + Historial → GPT-4

---

## 🚀 Ejecución

```bash
streamlit run main.py
```

[▶️ Ver demo en funcionamiento](demo.mp4)

---

## 🛠️ Personalización

En `main.py` ajustar:

```python
CHUNK_SIZE = 500       # Tamaño de fragmentos de texto
TEMPERATURE = 0.5      # Creatividad del modelo (0-1)
K_CONTEXT = 3          # Documentos relevantes a recuperar
```

---

## 🚨 Solución de Problemas

**Problema**: Error de autenticación con Pinecone  
**Solución**: Verificar que el índice tenga exactamente **768 dimensiones**

**Problema**: Respuestas genéricas  
**Solución**:  
- Aumentar `K_CONTEXT` a `5`  
- Revisar chunks en `split_text_with_langchain()`

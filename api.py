from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import get_settings
from src.dependencies import verify_api_key
from predict import MessageClassifier


classifier: MessageClassifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    settings = get_settings()
    print(f"Loading model from {settings.model_path}...")
    classifier = MessageClassifier(model_path=settings.model_path)
    print("Model loaded successfully")
    yield
    print("Shutting down...")


def create_app() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="""
## Messages Classification API

REST API untuk klasifikasi pesan customer service.

### Features
- Klasifikasi label: **issues** atau **search**
- Klasifikasi priority: **1** (high), **2** (medium), **3** (low)
- Multi-task learning dengan IndoBERT
- API Key authentication

### Authentication
Include `X-API-Key` header in all requests to `/api/v1/*` endpoints.
        """,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "Internal server error",
                "detail": str(exc) if settings.debug else None
            }
        )
    
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.api_title,
            "version": settings.api_version,
            "docs": "/docs",
            "health": "/health"
        }
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Check API health and model status (no auth required)"""
        return HealthResponse(
            status="healthy",
            model_loaded=classifier is not None
        )
    
    @app.post(
        "/api/v1/predict",
        response_model=PredictResponse,
        tags=["Prediction"],
        dependencies=[Depends(verify_api_key)]
    )
    async def predict(request: PredictRequest):
        """
        Klasifikasi pesan ke label dan priority.
        
        - **label**: 'issues' atau 'search'
        - **priority**: 1 (high), 2 (medium), atau 3 (low)
        
        Requires `X-API-Key` header.
        """
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        result = classifier.predict(request.text)
        
        return PredictResponse(
            success=True,
            message="Prediction successful",
            data=PredictionResult(
                label=result["label"],
                priority=result["priority"],
                label_confidence=round(result["label_confidence"], 4),
                priority_confidence=round(result["priority_confidence"], 4)
            )
        )
    
    return app


class PredictRequest(BaseModel):
    text: str
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "pesanan saya belum sampai sudah 3 hari"}
            ]
        }
    }


class PredictionResult(BaseModel):
    label: str
    priority: int
    label_confidence: float
    priority_confidence: float


class PredictResponse(BaseModel):
    success: bool
    message: str
    data: PredictionResult
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "message": "Prediction successful",
                    "data": {
                        "label": "issues",
                        "priority": 1,
                        "label_confidence": 0.95,
                        "priority_confidence": 0.82
                    }
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)

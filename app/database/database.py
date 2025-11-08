from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base # <--- Змінено тут
import os

# БД повинна лежати в КОРЕНЕВІЙ папці проекту, а не в 'app'
# Шлях 'sqlite:///./lab3.db'
SQLALCHEMY_DATABASE_URL = "sqlite:///./lab3.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base = declarative_base() # <--- Старий варіант
Base = declarative_base() # <--- Новий варіант

# Функція для отримання сесії БД
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
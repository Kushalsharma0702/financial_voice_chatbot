import os
import uuid
import logging
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer, DECIMAL, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timedelta
import json # For loading/dumping recent_payments in sample data
from contextlib import contextmanager


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@host:port/database_name") # Fallback for env var not set
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set. Please configure it in .env")

try:
    engine = create_engine(DATABASE_URL)
    logger.info("🏆 Database connected successfully!")
except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")
    raise

Base = declarative_base()
Session = sessionmaker(bind=engine)

# --- Models ---

class Customer(Base):
    __tablename__ = 'customer'
    customer_id = Column(String(20), primary_key=True)
    full_name = Column(String(100))
    phone_number = Column(String(15))
    email = Column(String(100))
    pan_number = Column(String(10))
    aadhaar_number = Column(String(20))
    kyc_status = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Customer(customer_id='{self.customer_id}', full_name='{self.full_name}')>"

class Loan(Base):
    __tablename__ = 'loan'
    loan_id = Column(String(20), primary_key=True)
    customer_id = Column(String(20), ForeignKey('customer.customer_id'))
    loan_type = Column(String(30))
    principal_amount = Column(DECIMAL(12,2))
    interest_rate = Column(DECIMAL(5,2))
    tenure_months = Column(Integer)
    start_date = Column(DateTime)
    status = Column(String(20))
    ifsc_code = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Loan(loan_id='{self.loan_id}', customer_id='{self.customer_id}', loan_type='{self.loan_type}')>"

class EMI(Base):
    __tablename__ = 'emi'
    emi_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    loan_id = Column(String(20), ForeignKey('loan.loan_id'))
    due_date = Column(DateTime)
    amount_due = Column(DECIMAL(10,2))
    amount_paid = Column(DECIMAL(10,2))
    payment_date = Column(DateTime)
    status = Column(String(20)) # e.g., 'Paid', 'Pending', 'Overdue'
    penalty_charged = Column(DECIMAL(10,2))
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return (f"<EMI(emi_id='{self.emi_id}', loan_id='{self.loan_id}', "
                f"due_date='{self.due_date}', amount_due='{self.amount_due}')>")

class CustomerAccount(Base):
    __tablename__ = 'customer_account'
    account_id = Column(String(20), primary_key=True)
    customer_id = Column(String(20), ForeignKey('customer.customer_id'))
    account_type = Column(String(20))
    balance = Column(DECIMAL(12,2))
    credit_limit = Column(DECIMAL(12,2))
    status = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<CustomerAccount(account_id='{self.account_id}', customer_id='{self.customer_id}')>"

class Transaction(Base):
    __tablename__ = 'transaction'
    transaction_id = Column(String(20), primary_key=True) # Could also be UUID
    account_id = Column(String(20), ForeignKey('customer_account.account_id'))
    customer_id = Column(String(20), ForeignKey('customer.customer_id'))
    account_type = Column(String(20)) # e.g., 'Savings', 'Credit Card'
    transaction_type = Column(String(30)) # e.g., 'Debit', 'Credit', 'Payment'
    amount = Column(DECIMAL(10,2))
    transaction_date = Column(DateTime, default=datetime.utcnow)
    description = Column(Text)

    def __repr__(self):
        return f"<Transaction(transaction_id='{self.transaction_id}', amount='{self.amount}', type='{self.transaction_type}')>"

class ClientInteraction(Base):
    __tablename__ = 'client_interaction'
    interaction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False)
    customer_id = Column(String(20), ForeignKey('customer.customer_id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    sender = Column(String(10), nullable=False) # 'user' or 'bot'
    message_text = Column(Text, nullable=False)
    intent = Column(String(50)) # e.g., 'emi', 'balance', 'loan', 'unclear'
    stage = Column(String(50)) # e.g., 'account_id_entry', 'otp_prompt', 'otp_verified', 'chat_message', 'intent_unclear', 'query_resolved'
    feedback_provided = Column(Boolean, default=False)
    feedback_positive = Column(Boolean)
    raw_response_data = Column(JSONB) # To store raw JSON response from external APIs (like Bedrock or database)
    embedding = Column(Vector(1536)) # For individual message embeddings if needed for advanced RAG

    def __repr__(self):
        return f"<ClientInteraction(session_id='{self.session_id}', sender='{self.sender}', intent='{self.intent}')>"

class RAGDocument(Base):
    __tablename__ = 'rag_document'
    document_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(String(20), ForeignKey('customer.customer_id'))
    document_text = Column(Text)
    embedding = Column(Vector(1024), nullable=True) # Ensure this matches the vector dimension from your embedding model
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<RAGDocument(document_id='{self.document_id}', customer_id='{self.customer_id}')>"

class OTP(Base):
    __tablename__ = 'otps'
    id = Column(Integer, primary_key=True)
    phone_number = Column(String(15), nullable=False)
    otp_code = Column(String(6), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"<OTP(phone_number='{self.phone_number}', otp_code='{self.otp_code}', expires_at='{self.expires_at}')>"

class UnresolvedChat(Base):
    __tablename__ = 'unresolved_chats'
    id = Column(Integer, primary_key=True)
    customer_id = Column(String, nullable=False)
    account_id = Column(String)
    session_id = Column(String, nullable=False) # String for CallSid or UUID string
    summary = Column(Text, nullable=False)
    embedding_vector = Column(Vector(1536)) # Assuming 1536 for general embeddings
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    resolved_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<UnresolvedChat(customer_id='{self.customer_id}', session_id='{self.session_id}', summary='{self.summary[:30]}...')>"


def create_tables():
    """Creates all defined tables in the database."""
    try:
        # For pgvector to work, you might need to enable the extension if not already done:
        # with engine.connect() as connection:
        #     connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        #     connection.commit()
        #     logger.info("Vector extension enabled (if not already).")
            
        Base.metadata.create_all(engine)
        logger.info("🏆 Tables created or verified successfully!")
    except Exception as e:
        logger.error(f"❌ Table creation failed: {e}", exc_info=True)
        raise

@contextmanager
def db_session():
    """Provides a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Database session error: {e}", exc_info=True)
        raise
    finally:
        session.close()

def setup_database():
    """Sets up the database by creating tables and refreshing sample data."""
    try:
        create_tables()
        refresh_sample_data()
        logger.info("Database setup completed successfully.")
    except Exception as e:
        logger.error(f"❌ Full database setup failed: {e}", exc_info=True)
        raise

def refresh_sample_data():
    logger.info("Refreshing EMI sample data...")
    try:
        with db_session() as session:
            # Clear existing sample data to ensure fresh start
            session.query(Customer).filter_by(account_id="CC62287740").delete()
            session.query(Loan).filter_by(customer_id="CID1000095").delete()
            session.query(EMI).filter_by(loan_id="LN54375877301289").delete()
            session.query(CustomerAccount).filter_by(customer_id="CID1000095").delete()
            session.query(OTP).filter_by(phone_number="+917417119014").delete()
            session.query(UnresolvedChat).delete() # Clear all unresolved chats
            session.query(ClientInteraction).filter_by(customer_id="CID1000095").delete()
            session.query(RAGDocument).filter_by(customer_id="CID1000095").delete()
            session.commit()
            logger.info("Cleared existing sample data.")

            # Add sample customer
            customer = Customer(
                customer_id="CID1000095",
                full_name="John Doe",
                phone_number="+917417119014",
                email="john.doe@example.com",
                pan_number="ABCDE1234F",
                aadhaar_number="123456789012",
                kyc_status="Verified"
            )
            session.add(customer)
            session.flush() # Flush to get customer_id before using in foreign key

            # Add sample loan
            loan = Loan(
                loan_id="LN54375877301289",
                customer_id=customer.customer_id,
                loan_type="Personal Loan",
                principal_amount=1380711.0,
                interest_rate=8.5,
                tenure_months=24,
                start_date=datetime(2024, 1, 1),
                status="Active",
                ifsc_code="SBIN0001234"
            )
            session.add(loan)
            session.flush()

            # Add sample EMI record
            monthly_emi_amount = 37440.31 # As per previous context
            emi_record = EMI(
                loan_id=loan.loan_id,
                due_date=datetime.utcnow() + timedelta(days=30), # Next month
                amount_due=monthly_emi_amount,
                amount_paid=0.0,
                payment_date=None,
                status="Pending",
                penalty_charged=0.0
            )
            session.add(emi_record)
            session.flush()

            # Add sample customer account
            customer_account = CustomerAccount(
                account_id="CC62287740",
                customer_id=customer.customer_id,
                account_type="Savings",
                balance=150000.00,
                credit_limit=0.00,
                status="Active"
            )
            session.add(customer_account)
            session.flush()

            # Add sample transaction
            transaction = Transaction(
                transaction_id="TXN123456789",
                account_id=customer_account.account_id,
                customer_id=customer.customer_id,
                account_type="Savings",
                transaction_type="Deposit",
                amount=50000.00,
                transaction_date=datetime.utcnow(),
                description="Initial deposit"
            )
            session.add(transaction)
            session.flush()

            # Add a sample RAG document (e.g., policy or general info)
            rag_doc = RAGDocument(
                customer_id=customer.customer_id, # Can be linked to a customer or general
                document_text="Our personal loan interest rates range from 8.5% to 15% depending on credit score and tenure. Loan amounts can be up to 20 lakhs. For more details, please visit our website or contact support.",
                embedding=[random.random() for _ in range(1024)] # Dummy embedding
            )
            session.add(rag_doc)
            session.flush()

            # Manually commit all changes at the end of the block
            session.commit()
            logger.info("Sample data refreshed successfully.")
    except Exception as e:
        session.rollback() # Ensure rollback on error
        logger.error(f"❌ Error refreshing sample data: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("Attempting to connect to database and create tables...")
    setup_database()
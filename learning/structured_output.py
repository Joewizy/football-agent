from pydantic import BaseModel
from langchain_ollama import ChatOllama


class Person(BaseModel):
    name: str
    age: int
    occupation: str

class CryptoToken(BaseModel):
    name: str
    symbol: str
    blockchain: str
    consensus: str

class Rate(BaseModel):
    amount: float
    currency: str


llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

structured_llm = llm.with_structured_output(Person)
structured_llm_crypto = llm.with_structured_output(CryptoToken)
structured_llm_rate = llm.with_structured_output(Rate)

response = structured_llm.invoke(
    """
    Sarah perfromed a surgery last week.
    She recently celebrated her 42nd birthday.
    """
)

crypto_response = structured_llm_crypto.invoke(
    """
    Ethereum uses ETH as its native token and operates on the Ethereum blockchain using Proof of Stake.
    """
)

rate_response = structured_llm_rate.invoke(
    """
    The price of one dollar in naira today is 100 naira.
    """
)

print(response)
print(crypto_response)
print(rate_response)
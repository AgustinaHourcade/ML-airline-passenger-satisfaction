from pydantic import BaseModel, Field
from typing import Optional

class Phase1Features(BaseModel):
    # Demographics & Flight basic info
    Age: int = Field(..., ge=1, le=120, description="Age of the passenger")
    Gender: str = Field(..., description="Gender: 'Male' or 'Female'")
    Customer_Type: str = Field(..., alias="Customer Type", description="'Loyal Customer' or 'disloyal Customer'")
    Type_of_Travel: str = Field(..., alias="Type of Travel", description="'Business travel' or 'Personal Travel'")
    Class: str = Field(..., description="'Business', 'Eco', or 'Eco Plus'")
    Flight_Distance: int = Field(..., alias="Flight Distance", ge=1, description="Flight distance in miles")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Age": 35,
                "Gender": "Female",
                "Customer Type": "Loyal Customer",
                "Type of Travel": "Business travel",
                "Class": "Business",
                "Flight Distance": 1500
            }
        }

class Phase2Features(Phase1Features):
    # Post-flight survey scores (Ordinal 0-5)
    Seat_comfort: int = Field(..., alias="Seat comfort", ge=0, le=5)
    Departure_Arrival_time_convenient: int = Field(..., alias="Departure/Arrival time convenient", ge=0, le=5)
    Food_and_drink: int = Field(..., alias="Food and drink", ge=0, le=5)
    Gate_location: int = Field(..., alias="Gate location", ge=0, le=5)
    Inflight_wifi_service: int = Field(..., alias="Inflight wifi service", ge=0, le=5)
    Inflight_entertainment: int = Field(..., alias="Inflight entertainment", ge=0, le=5)
    Online_support: int = Field(..., alias="Online support", ge=0, le=5)
    Ease_of_Online_booking: int = Field(..., alias="Ease of Online booking", ge=0, le=5)
    On_board_service: int = Field(..., alias="On-board service", ge=0, le=5)
    Leg_room_service: int = Field(..., alias="Leg room service", ge=0, le=5)
    Baggage_handling: int = Field(..., alias="Baggage handling", ge=0, le=5)
    Checkin_service: int = Field(..., alias="Checkin service", ge=0, le=5)
    Cleanliness: int = Field(..., alias="Cleanliness", ge=0, le=5)
    Online_boarding: int = Field(..., alias="Online boarding", ge=0, le=5)
    
    # Numeric post-flight
    Arrival_Delay_in_Minutes: Optional[float] = Field(None, alias="Arrival Delay in Minutes", ge=0)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Age": 35,
                "Gender": "Female",
                "Customer Type": "Loyal Customer",
                "Type of Travel": "Business travel",
                "Class": "Business",
                "Flight Distance": 1500,
                "Seat comfort": 4,
                "Departure/Arrival time convenient": 5,
                "Food and drink": 4,
                "Gate location": 3,
                "Inflight wifi service": 5,
                "Inflight entertainment": 4,
                "Online support": 5,
                "Ease of Online booking": 5,
                "On-board service": 4,
                "Leg room service": 4,
                "Baggage handling": 4,
                "Checkin service": 5,
                "Cleanliness": 4,
                "Online boarding": 5,
                "Arrival Delay in Minutes": 10
            }
        }

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for dissatisfied, 1 for satisfied")
    probability: float = Field(..., description="Probability of being satisfied")
    phase: int = Field(..., description="Which model phase was used (1 or 2)")

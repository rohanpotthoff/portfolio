import pytz
import datetime
import streamlit as st
import requests
import json
from typing import Tuple, Optional, Dict, List

class TimezoneHandler:
    """
    Handles timezone detection, conversion, and market hours calculation
    for the portfolio management system.
    """
    
    # Market timezone mapping
    MARKET_TIMEZONES = {
        "US": pytz.timezone("America/New_York"),
        "Europe": pytz.timezone("Europe/London"),
        "Asia": pytz.timezone("Asia/Tokyo"),
        "Australia": pytz.timezone("Australia/Sydney")
    }
    
    # Market hours by region (24-hour format)
    MARKET_HOURS = {
        "US": {"open": (9, 30), "close": (16, 0)},
        "Europe": {"open": (8, 0), "close": (16, 30)},
        "Asia": {"open": (9, 0), "close": (15, 0)},
        "Australia": {"open": (10, 0), "close": (16, 0)}
    }
    
    # US Market Holidays
    US_HOLIDAYS = [
        # 2024 US Market Holidays
        datetime.date(2024, 1, 1),    # New Year's Day
        datetime.date(2024, 1, 15),   # Martin Luther King Jr. Day
        datetime.date(2024, 2, 19),   # Presidents' Day
        datetime.date(2024, 3, 29),   # Good Friday
        datetime.date(2024, 5, 27),   # Memorial Day
        datetime.date(2024, 6, 19),   # Juneteenth
        datetime.date(2024, 7, 4),    # Independence Day
        datetime.date(2024, 9, 2),    # Labor Day
        datetime.date(2024, 11, 28),  # Thanksgiving Day
        datetime.date(2024, 12, 25),  # Christmas Day
        # 2025 US Market Holidays
        datetime.date(2025, 1, 1),    # New Year's Day
        datetime.date(2025, 1, 20),   # Martin Luther King Jr. Day
        datetime.date(2025, 2, 17),   # Presidents' Day
        datetime.date(2025, 4, 18),   # Good Friday
        datetime.date(2025, 5, 26),   # Memorial Day
        datetime.date(2025, 6, 19),   # Juneteenth
        datetime.date(2025, 7, 4),    # Independence Day
        datetime.date(2025, 9, 1),    # Labor Day
        datetime.date(2025, 11, 27),  # Thanksgiving Day
        datetime.date(2025, 12, 25),  # Christmas Day
    ]
    
    # European Market Holidays (major ones)
    EUROPE_HOLIDAYS = [
        # 2024 European Market Holidays (major ones)
        datetime.date(2024, 1, 1),    # New Year's Day
        datetime.date(2024, 3, 29),   # Good Friday
        datetime.date(2024, 4, 1),    # Easter Monday
        datetime.date(2024, 5, 1),    # Labor Day
        datetime.date(2024, 12, 25),  # Christmas Day
        datetime.date(2024, 12, 26),  # Boxing Day
        # 2025 European Market Holidays (major ones)
        datetime.date(2025, 1, 1),    # New Year's Day
        datetime.date(2025, 4, 18),   # Good Friday
        datetime.date(2025, 4, 21),   # Easter Monday
        datetime.date(2025, 5, 1),    # Labor Day
        datetime.date(2025, 12, 25),  # Christmas Day
        datetime.date(2025, 12, 26),  # Boxing Day
    ]
    
    def __init__(self):
        """Initialize the timezone handler with user's local timezone"""
        self.user_timezone = self._detect_user_timezone()
        self.default_market = "US"  # Default to US markets
    
    def _detect_user_timezone(self) -> pytz.timezone:
        """
        Detect user's timezone based on browser information or IP geolocation.
        Falls back to America/New_York if detection fails.
        """
        try:
            # Try to get timezone from session state if already detected
            if 'user_timezone' in st.session_state:
                return pytz.timezone(st.session_state['user_timezone'])
            
            # Try to get timezone from IP geolocation
            response = requests.get('https://ipapi.co/json/', timeout=3)
            if response.status_code == 200:
                data = response.json()
                if 'timezone' in data:
                    timezone_str = data['timezone']
                    st.session_state['user_timezone'] = timezone_str
                    return pytz.timezone(timezone_str)
        except Exception:
            # Silently fail and use default
            pass
        
        # Default to Eastern Time
        return pytz.timezone("America/New_York")
    
    def get_user_timezone(self) -> pytz.timezone:
        """Get the user's detected timezone"""
        return self.user_timezone
    
    def get_market_timezone(self, market: str = None) -> pytz.timezone:
        """Get the timezone for a specific market"""
        if not market:
            market = self.default_market
        return self.MARKET_TIMEZONES.get(market, self.MARKET_TIMEZONES["US"])
    
    def now(self) -> datetime.datetime:
        """Get current datetime in user's timezone"""
        return datetime.datetime.now(self.user_timezone)
    
    def format_datetime(self, dt: datetime.datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime for display in user's timezone"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(self.user_timezone).strftime(fmt)
    
    def convert_to_market_time(self, dt: datetime.datetime, market: str = None) -> datetime.datetime:
        """Convert datetime to market timezone"""
        if not market:
            market = self.default_market
            
        market_tz = self.get_market_timezone(market)
        
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
            
        return dt.astimezone(market_tz)
    
    def get_market_hours(self, market: str = None, date: datetime.date = None) -> Tuple[datetime.datetime, datetime.datetime]:
        """
        Get market open and close times for a specific date.
        Handles weekends and holidays.
        """
        if not market:
            market = self.default_market
            
        if not date:
            date = self.now().date()
            
        market_tz = self.get_market_timezone(market)
        
        # Check if date is a weekend
        is_weekend = date.weekday() >= 5  # Saturday or Sunday
        
        # Check if date is a holiday
        is_holiday = False
        if market == "US":
            is_holiday = date in self.US_HOLIDAYS
        elif market == "Europe":
            is_holiday = date in self.EUROPE_HOLIDAYS
            
        if is_weekend or is_holiday:
            # Find the next trading day
            next_date = date
            while True:
                next_date += datetime.timedelta(days=1)
                next_weekend = next_date.weekday() >= 5
                next_holiday = False
                if market == "US":
                    next_holiday = next_date in self.US_HOLIDAYS
                elif market == "Europe":
                    next_holiday = next_date in self.EUROPE_HOLIDAYS
                    
                if not (next_weekend or next_holiday):
                    break
                    
            date = next_date
            
        # Get market hours for the date
        hours = self.MARKET_HOURS.get(market, self.MARKET_HOURS["US"])
        open_hour, open_minute = hours["open"]
        close_hour, close_minute = hours["close"]
        
        # Create datetime objects in market timezone
        open_time = datetime.datetime(
            date.year, date.month, date.day, 
            open_hour, open_minute, 0, 0, 
            tzinfo=market_tz
        )
        
        close_time = datetime.datetime(
            date.year, date.month, date.day, 
            close_hour, close_minute, 0, 0, 
            tzinfo=market_tz
        )
        
        return open_time, close_time
    
    def is_market_open(self, market: str = None) -> bool:
        """Check if a specific market is currently open"""
        if not market:
            market = self.default_market
            
        now = self.now()
        open_time, close_time = self.get_market_hours(market, now.date())
        
        # Convert to user timezone for comparison
        open_time = open_time.astimezone(self.user_timezone)
        close_time = close_time.astimezone(self.user_timezone)
        
        return open_time <= now <= close_time
    
    def get_time_to_market_open(self, market: str = None) -> Optional[datetime.timedelta]:
        """Get time until market opens"""
        if not market:
            market = self.default_market
            
        now = self.now()
        open_time, _ = self.get_market_hours(market, now.date())
        
        # Convert to user timezone for comparison
        open_time = open_time.astimezone(self.user_timezone)
        
        if now < open_time:
            return open_time - now
        return None
    
    def get_time_to_market_close(self, market: str = None) -> Optional[datetime.timedelta]:
        """Get time until market closes"""
        if not market:
            market = self.default_market
            
        now = self.now()
        _, close_time = self.get_market_hours(market, now.date())
        
        # Convert to user timezone for comparison
        close_time = close_time.astimezone(self.user_timezone)
        
        if now < close_time:
            return close_time - now
        return None
    
    def get_market_status_display(self, market: str = None) -> str:
        """Get a display string for market status"""
        if not market:
            market = self.default_market
            
        if self.is_market_open(market):
            time_to_close = self.get_time_to_market_close(market)
            hours, remainder = divmod(time_to_close.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            return f"🟢 Market Open (Closes in {hours}h {minutes}m)"
        else:
            time_to_open = self.get_time_to_market_open(market)
            if time_to_open:
                hours, remainder = divmod(time_to_open.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                return f"🔴 Market Closed (Opens in {hours}h {minutes}m)"
            else:
                # Market closed for the day
                open_time, _ = self.get_market_hours(market, self.now().date() + datetime.timedelta(days=1))
                open_time = open_time.astimezone(self.user_timezone)
                return f"🔴 Market Closed (Opens {open_time.strftime('%a %H:%M')})"

# Create a singleton instance
timezone_handler = TimezoneHandler()
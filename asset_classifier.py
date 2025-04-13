import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
import re

class AssetClassifier:
    """
    Handles automatic classification of financial assets based on ticker characteristics
    and metadata from various sources.
    """
    
    # Asset class definitions
    ASSET_CLASSES = {
        "Money Market": "Cash equivalents and short-term, highly liquid investments",
        "Bond": "Fixed income securities representing loans to an entity",
        "Stock": "Equity securities representing ownership in a corporation",
        "ETF": "Exchange-traded funds that track indexes, commodities, or baskets of assets",
        "REIT": "Real Estate Investment Trusts that own or finance income-producing real estate",
        "Commodity": "Basic goods used in commerce that are interchangeable with other goods of the same type",
        "Cryptocurrency": "Digital or virtual currencies that use cryptography for security",
        "Alternative": "Investments that fall outside the traditional asset classes",
        "Mutual Fund": "Investment vehicles made up of a pool of funds from many investors",
        "Index": "Market indexes that track the performance of a specific market or sector",
        "Option": "Contracts giving the right to buy or sell an asset at a specific price",
        "Futures": "Contracts to buy or sell an asset at a future date at an agreed-upon price",
        "Forex": "Foreign exchange currency pairs",
        "Unknown": "Asset class could not be determined"
    }
    
    # Sector mappings
    SECTORS = {
        "Technology": "Companies that primarily develop, manufacture, or distribute technology products",
        "Healthcare": "Companies involved in medical services, equipment, or pharmaceuticals",
        "Financial": "Companies that provide financial services to commercial and retail customers",
        "Consumer Cyclical": "Companies that tend to be the most sensitive to economic cycles",
        "Consumer Defensive": "Companies that are less sensitive to economic cycles",
        "Industrials": "Companies that manufacture and distribute capital goods",
        "Basic Materials": "Companies involved in the discovery, development, and processing of raw materials",
        "Energy": "Companies involved in the exploration, production, and distribution of energy",
        "Utilities": "Companies that provide basic amenities, such as water, sewage, and electricity",
        "Real Estate": "Companies involved in real estate development, operations, and services",
        "Communication Services": "Companies that provide communication services",
        "Unknown": "Sector could not be determined"
    }
    
    # Patterns for identifying asset classes
    PATTERNS = {
        "Money Market": [r"MM$", r"XX$", r"CASH$"],
        "Bond": [r"BOND", r"TREASURY", r"GOVT", r"CORP", r"-B$"],
        "ETF": [r"ETF$", r"^SPY$", r"^QQQ$", r"^DIA$", r"^IWM$", r"^EEM$", r"^VTI$", r"^CLOU$", r"^KBE$", r"^QQQJ$", r"^SIXG$"],
        "REIT": [r"REIT", r"^VNQ$", r"^IYR$", r"^SPG$", r"^O$", r"^AMT$"],
        "Commodity": [r"GOLD", r"SILVER", r"OIL", r"GAS", r"^GLD$", r"^SLV$", r"^USO$", r"^UNG$"],
        "Cryptocurrency": [r"-USD$", r"-EUR$", r"^BTC", r"^ETH", r"^DOGE", r"^XRP"],
        "Index": [r"^\^", r"INDEX"],
        "Option": [r"\d+[CP]\d+$"],
        "Futures": [r"\w+\d+$"],
        "Forex": [r"^[A-Z]{3}[A-Z]{3}=X$"]
    }
    
    # Common ETF providers
    ETF_PROVIDERS = [
        "VANGUARD", "ISHARES", "SPDR", "SCHWAB", "INVESCO", "PROSHARES", "DIREXION",
        "FIRST TRUST", "WISDOMTREE", "GLOBAL X", "ARK", "VANECK", "JPMORGAN"
    ]
    
    # Common money market fund identifiers
    MONEY_MARKET_TICKERS = ["WMPXX", "FNSXX", "VMFXX", "SPAXX", "SWVXX", "VMMXX"]
    
    def __init__(self):
        """Initialize the asset classifier with cache for performance"""
        self.cache = {}
        self.etf_list = set()
        self.reit_list = set()
        self.load_known_assets()
    
    def load_known_assets(self):
        """Load known ETFs and REITs from predefined lists"""
        # Common ETFs
        self.etf_list = {
            "SPY", "QQQ", "DIA", "IWM", "EEM", "VTI", "CLOU", "KBE", "QQQJ", "SIXG",
            "VOO", "VEA", "VWO", "BND", "AGG", "VIG", "VYM", "VUG", "VTV", "VNQ",
            "IJH", "IJR", "IVV", "IWF", "IWD", "EFA", "LQD", "TLT", "GLD", "SLV",
            "XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE"
        }
        
        # Common REITs
        self.reit_list = {
            "SPG", "O", "AMT", "PLD", "CCI", "EQIX", "DLR", "AVB", "EQR", "ESS",
            "PSA", "WELL", "VTR", "BXP", "ARE", "HST", "UDR", "EXR", "MAA", "REG"
        }
    
    def classify(self, ticker: str, info: Optional[Dict] = None) -> Dict:
        """
        Classify a financial asset based on its ticker and available information.
        
        Args:
            ticker: The ticker symbol of the asset
            info: Optional dictionary with additional information about the asset
                 (typically from yfinance Ticker.info)
                 
        Returns:
            Dictionary with classification information:
            {
                "asset_class": str,
                "sector": str,
                "industry": str,
                "description": str,
                "confidence": float
            }
        """
        # Check cache first
        if ticker in self.cache:
            return self.cache[ticker]
        
        # Initialize result
        result = {
            "asset_class": "Unknown",
            "sector": "Unknown",
            "industry": "Unknown",
            "description": "",
            "confidence": 0.0
        }
        
        # Try to classify based on ticker pattern first
        confidence = 0.0
        asset_class = self._classify_by_pattern(ticker)
        if asset_class != "Unknown":
            result["asset_class"] = asset_class
            confidence = 0.7  # Pattern matching has good confidence
        
        # If we have additional info, use it to refine classification
        if info:
            refined_result = self._classify_with_info(ticker, info)
            
            # Only update if we have better confidence
            if refined_result["confidence"] > confidence:
                result = refined_result
        
        # If still unknown, try to fetch info from yfinance
        if result["asset_class"] == "Unknown" or result["confidence"] < 0.8:
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                refined_result = self._classify_with_info(ticker, info)
                
                # Only update if we have better confidence
                if refined_result["confidence"] > result["confidence"]:
                    result = refined_result
            except Exception:
                # Silently fail and use what we have
                pass
        
        # Cache the result
        self.cache[ticker] = result
        return result
    
    def _classify_by_pattern(self, ticker: str) -> str:
        """Classify asset based on ticker pattern matching"""
        # Check for known ETFs and REITs first
        if ticker in self.etf_list:
            return "ETF"
        if ticker in self.reit_list:
            return "REIT"
        if ticker in self.MONEY_MARKET_TICKERS:
            return "Money Market"
        
        # Check patterns
        for asset_class, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, ticker):
                    return asset_class
        
        # Default to stock if no pattern matches
        return "Stock"
    
    def _classify_with_info(self, ticker: str, info: Dict) -> Dict:
        """Classify asset using additional information"""
        result = {
            "asset_class": "Unknown",
            "sector": "Unknown",
            "industry": "Unknown",
            "description": "",
            "confidence": 0.0
        }
        
        # Extract useful fields
        quote_type = info.get("quoteType", "").lower()
        security_type = info.get("securityType", "").lower()
        instrument_type = info.get("instrumentType", "").lower()
        fund_family = info.get("fundFamily", "")
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        description = info.get("longBusinessSummary", "")
        
        # Determine asset class
        if quote_type in ["cryptocurrency", "crypto"]:
            result["asset_class"] = "Cryptocurrency"
            result["confidence"] = 0.95
        elif quote_type == "etf" or security_type == "etf" or instrument_type == "etf":
            result["asset_class"] = "ETF"
            result["confidence"] = 0.95
        elif quote_type == "mutualfund" or security_type == "mutualfund":
            result["asset_class"] = "Mutual Fund"
            result["confidence"] = 0.95
        elif quote_type == "index":
            result["asset_class"] = "Index"
            result["confidence"] = 0.95
        elif quote_type in ["equity", "stock"]:
            # Check if it's a REIT
            if "reit" in description.lower() or "real estate investment trust" in description.lower():
                result["asset_class"] = "REIT"
                result["confidence"] = 0.9
            else:
                result["asset_class"] = "Stock"
                result["confidence"] = 0.9
        elif quote_type == "option":
            result["asset_class"] = "Option"
            result["confidence"] = 0.95
        elif quote_type == "future":
            result["asset_class"] = "Futures"
            result["confidence"] = 0.95
        elif quote_type == "currency":
            result["asset_class"] = "Forex"
            result["confidence"] = 0.95
        elif "bond" in security_type or "bond" in description.lower():
            result["asset_class"] = "Bond"
            result["confidence"] = 0.9
        elif fund_family:
            # Check if it's from a known ETF provider
            for provider in self.ETF_PROVIDERS:
                if provider.lower() in fund_family.lower():
                    result["asset_class"] = "ETF"
                    result["confidence"] = 0.85
                    break
            
            # If still unknown, default to Mutual Fund
            if result["asset_class"] == "Unknown":
                result["asset_class"] = "Mutual Fund"
                result["confidence"] = 0.8
        
        # If still unknown, use pattern matching as fallback
        if result["asset_class"] == "Unknown":
            asset_class = self._classify_by_pattern(ticker)
            result["asset_class"] = asset_class
            result["confidence"] = 0.7
        
        # Set sector and industry if available
        if sector:
            result["sector"] = sector
        if industry:
            result["industry"] = industry
        if description:
            result["description"] = description[:200] + "..." if len(description) > 200 else description
        
        return result
    
    def get_asset_class_description(self, asset_class: str) -> str:
        """Get the description for an asset class"""
        return self.ASSET_CLASSES.get(asset_class, "Unknown asset class")
    
    def get_sector_description(self, sector: str) -> str:
        """Get the description for a sector"""
        return self.SECTORS.get(sector, "Unknown sector")
    
    def get_all_asset_classes(self) -> Dict[str, str]:
        """Get all available asset classes with descriptions"""
        return self.ASSET_CLASSES
    
    def get_all_sectors(self) -> Dict[str, str]:
        """Get all available sectors with descriptions"""
        return self.SECTORS
    
    def clear_cache(self):
        """Clear the classification cache"""
        self.cache = {}

# Create a singleton instance
asset_classifier = AssetClassifier()


import time
from collections import defaultdict
from typing import Dict, Tuple
import asyncio
from fastapi import Request
from fastapi.responses import JSONResponse


class RateLimiter:
    """Simple rate limiter using sliding window"""
    
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # seconds
        self.requests: Dict[str, list] = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def is_rate_limited(self, client_id: str) -> Tuple[bool, Dict]:
        """Check if client is rate limited"""
        async with self.lock:
            current_time = time.time()
            
            # Clean old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if current_time - req_time < self.window_size
            ]
            
            # Check rate limit
            if len(self.requests[client_id]) >= self.requests_per_minute:
                return True, {
                    "limit": self.requests_per_minute,
                    "remaining": 0,
                    "reset_time": int(min(self.requests[client_id]) + self.window_size)
                }
            
            # Add current request
            self.requests[client_id].append(current_time)
            
            return False, {
                "limit": self.requests_per_minute,
                "remaining": self.requests_per_minute - len(self.requests[client_id]),
                "reset_time": int(current_time + self.window_size)
            }


# Global rate limiter instance
_limiter = RateLimiter(requests_per_minute=100)


async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting middleware for FastAPI
    """
    # Get client identifier (use IP address)
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    is_limited, rate_info = await _limiter.is_rate_limited(client_ip)
    
    if is_limited:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too Many Requests",
                "message": f"Rate limit exceeded. {_limiter.requests_per_minute} requests per minute allowed.",
                "retry_after": rate_info["reset_time"] - int(time.time())
            },
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": str(rate_info["remaining"]),
                "X-RateLimit-Reset": str(rate_info["reset_time"]),
                "Retry-After": str(rate_info["reset_time"] - int(time.time()))
            }
        )
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers to successful responses
    response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(rate_info["reset_time"])
    
    return response

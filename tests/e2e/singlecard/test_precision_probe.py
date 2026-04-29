# SPDX-License-Identifier: Apache-2.0
"""Precision probe: test all-mode vs align-mode across diverse prompts.

Checks whether all-mode produces correct answers and stops at EOS.
Uses 3 different prefix types (table, article, code), each long enough
for multi-block (3+ blocks at block_size=1024).

Run:
    pytest tests/e2e/singlecard/test_precision_probe.py::test_precision_probe -v -s
"""

import os
import pytest
from tests.e2e.conftest import VllmRunner

MODEL = "/data/Qwen3.5-9B"
MAX_TOKENS = 50

# ═══════════════════════════════════════════════════════════════════════
# PREFIX 1: Product inventory table (100 rows, ~3500 tokens)
# ═══════════════════════════════════════════════════════════════════════

_TBL_HEADER = (
    "You are an inventory analyst. Below is a product database.\n"
    "# Products\n\n"
    "| SKU    | Product Name                    | Price  | Stock | Category    | Brand | Rating | Color  | Warehouse    | Added      |\n"
    "|--------|---------------------------------|--------|-------|-------------|-------|--------|--------|--------------|------------|\n"
)

# fmt: off
# ruff: noqa: E501
_TBL_ROWS = """\
| P0001 | Wireless Noise-Cancel Earbuds   | 129.99 |   342 | Electronics | Sony  | 4.5  | Black  | West-01      | 2024-01-15 |
| P0002 | Organic Cotton T-Shirt Large    |  24.50 |   891 | Clothing    | Zara  | 4.2  | White  | East-03      | 2024-01-18 |
| P0003 | Premium Dark Chocolate Bar 200g |   8.99 |  1205 | Food        | Lindt | 4.8  | Brown  | Central-02   | 2024-01-20 |
| P0004 | Carbon Fiber Tennis Racket Pro  | 189.00 |   156 | Sports      | Wilson| 4.6  | Red    | South-04     | 2024-01-22 |
| P0005 | Bestseller Mystery Novel 2024   |  14.99 |  2340 | Books       | Pengn | 4.3  | Blue   | East-03      | 2024-01-25 |
| P0006 | Ceramic Pour-Over Coffee Set    |  45.00 |   478 | Home        | Hario | 4.7  | Gray   | West-01      | 2024-02-01 |
| P0007 | Building Blocks 500pc Mega Set  |  34.99 |   623 | Toys        | Lego  | 4.9  | Multi  | North-05     | 2024-02-03 |
| P0008 | Solar Powered Garden Lamp 4pk   |  29.99 |   367 | Garden      | Solux | 4.1  | Green  | South-04     | 2024-02-05 |
| P0009 | Stainless Steel Water Bottle 1L |  19.95 |  1543 | Home        | Hydro | 4.4  | Silver | Central-02   | 2024-02-08 |
| P0010 | Yoga Mat Premium 6mm Thick      |  39.99 |   289 | Sports      | Gaiam | 4.6  | Purple | West-01      | 2024-02-10 |
| P0011 | Bluetooth Speaker Waterproof    |  59.99 |   445 | Electronics | JBL   | 4.3  | Blue   | East-03      | 2024-02-12 |
| P0012 | Merino Wool Sweater Medium      |  79.00 |   312 | Clothing    | Uniqo | 4.5  | Navy   | North-05     | 2024-02-15 |
| P0013 | Assorted Herbal Tea Collection  |  12.50 |  1876 | Food        | Twnng | 4.6  | Green  | Central-02   | 2024-02-18 |
| P0014 | Adjustable Dumbbell Set 25kg    | 149.00 |   201 | Sports      | Bowfx | 4.4  | Black  | South-04     | 2024-02-20 |
| P0015 | Science Fiction Anthology 2024  |  18.99 |  1567 | Books       | Orbit | 4.7  | Red    | East-03      | 2024-02-22 |
| P0016 | Memory Foam Pillow Standard     |  49.99 |   534 | Home        | Tempr | 4.8  | White  | West-01      | 2024-02-25 |
| P0017 | Remote Control Drone with Cam   |  99.99 |   178 | Toys        | DJI   | 4.2  | Gray   | North-05     | 2024-02-28 |
| P0018 | Pruning Shears Professional     |  22.99 |   456 | Garden      | Fiskr | 4.5  | Orange | South-04     | 2024-03-01 |
| P0019 | USB-C Hub 7-in-1 Adapter        |  35.99 |   890 | Electronics | Anker | 4.6  | Silver | Central-02   | 2024-03-03 |
| P0020 | Linen Dress Shirt Slim Fit      |  55.00 |   267 | Clothing    | CKlein| 4.3  | LtBlue | East-03      | 2024-03-05 |
| P0021 | Organic Granola Honey Almond    |   9.49 |  2134 | Food        | Natrs | 4.5  | Brown  | West-01      | 2024-03-08 |
| P0022 | Resistance Bands Set of 5       |  16.99 |   712 | Sports      | Thera | 4.7  | Multi  | North-05     | 2024-03-10 |
| P0023 | Historical Fiction Epic Novel   |  16.50 |  1890 | Books       | HColns| 4.4  | Gold   | South-04     | 2024-03-12 |
| P0024 | Cast Iron Dutch Oven 6 Quart    |  64.99 |   345 | Home        | Lodge | 4.9  | Black  | Central-02   | 2024-03-15 |
| P0025 | Wooden Train Set 80 Pieces      |  44.99 |   398 | Toys        | Brio  | 4.8  | Multi  | East-03      | 2024-03-18 |
| P0026 | Drip Irrigation Starter Kit     |  38.99 |   234 | Garden      | Rainb | 4.2  | Green  | West-01      | 2024-03-20 |
| P0027 | Noise Cancel Headphones OverEar | 199.99 |   267 | Electronics | Bose  | 4.7  | Black  | North-05     | 2024-03-22 |
| P0028 | Fleece Zip-Up Jacket Women      |  42.00 |   456 | Clothing    | NthFc | 4.4  | Pink   | South-04     | 2024-03-25 |
| P0029 | Cold Brew Coffee Concentrate 1L |  15.99 |  1234 | Food        | Stmpn | 4.3  | Brown  | Central-02   | 2024-03-28 |
| P0030 | Folding Camping Chair Deluxe    |  54.99 |   189 | Sports      | Colmn | 4.5  | Green  | East-03      | 2024-04-01 |
| P0031 | Cookbook Mediterranean Kitchen   |  28.99 |  1456 | Books       | Phaid | 4.6  | Red    | West-01      | 2024-04-03 |
| P0032 | Robot Vacuum with Mapping       | 299.99 |   123 | Home        | Roomb | 4.1  | White  | North-05     | 2024-04-05 |
| P0033 | Plush Dinosaur Collection 6pk   |  26.99 |   567 | Toys        | Jelct | 4.9  | Multi  | South-04     | 2024-04-08 |
| P0034 | Raised Garden Bed Cedar 4x8ft   |  89.99 |    98 | Garden      | Grnsr | 4.3  | Brown  | Central-02   | 2024-04-10 |
| P0035 | Smart Watch Fitness Tracker     | 159.99 |   334 | Electronics | Fitbt | 4.4  | Black  | East-03      | 2024-04-12 |
| P0036 | Silk Blend Scarf Paisley Print  |  32.00 |   389 | Clothing    | Burbn | 4.7  | Red    | West-01      | 2024-04-15 |
| P0037 | Organic Olive Oil Extra Virgin  |  18.99 |  1678 | Food        | Berli | 4.8  | Gold   | North-05     | 2024-04-18 |
| P0038 | Inflatable Stand-Up Paddleboard | 279.99 |    76 | Sports      | AqMrn | 4.2  | Blue   | South-04     | 2024-04-20 |
| P0039 | Poetry Anthology Modern Voices  |  13.99 |  1234 | Books       | Faber | 4.5  | Purple | Central-02   | 2024-04-22 |
| P0040 | Electric Kettle Variable Temp   |  44.99 |   412 | Home        | Felws | 4.6  | Steel  | East-03      | 2024-04-25 |
| P0041 | Magnetic Tile Building Set 100  |  49.99 |   278 | Toys        | Magna | 4.8  | Multi  | West-01      | 2024-04-28 |
| P0042 | Compost Bin Tumbler 45 Gallon   |  74.99 |   145 | Garden      | Liftn | 4.1  | Black  | North-05     | 2024-05-01 |
| P0043 | Mechanical Keyboard RGB TKL     |  89.99 |   534 | Electronics | Corsr | 4.5  | Black  | South-04     | 2024-05-03 |
| P0044 | Puffer Vest Lightweight Down    |  68.00 |   234 | Clothing    | Patgn | 4.6  | Olive  | Central-02   | 2024-05-05 |
| P0045 | Artisan Sourdough Bread Mix 3pk |  11.99 |  1567 | Food        | KngAr | 4.4  | Wheat  | East-03      | 2024-05-08 |
| P0046 | Hiking Boots Waterproof Mid     | 129.99 |   189 | Sports      | Salmn | 4.7  | Brown  | West-01      | 2024-05-10 |
| P0047 | Graphic Novel Sci-Fi Saga Vol1  |  22.99 |  1890 | Books       | Image | 4.3  | Blue   | North-05     | 2024-05-12 |
| P0048 | Air Purifier HEPA Large Room    | 179.99 |   156 | Home        | Dyson | 4.5  | White  | South-04     | 2024-05-15 |
| P0049 | Chemistry Experiment Kit Kids   |  36.99 |   345 | Toys        | NatGo | 4.7  | Multi  | Central-02   | 2024-05-18 |
| P0050 | Hanging Planter Macrame Set 3pc |  24.99 |   567 | Garden      | Blmfy | 4.4  | Cream  | East-03      | 2024-05-20 |
| P0051 | Portable SSD 2TB USB-C          | 119.99 |   412 | Electronics | Samsg | 4.8  | Black  | West-01      | 2024-05-22 |
| P0052 | Running Shoes Lightweight Mesh  |  95.00 |   298 | Clothing    | Nike  | 4.3  | White  | North-05     | 2024-05-25 |
| P0053 | Matcha Powder Ceremonial Grade  |  24.99 |   890 | Food        | Ippdo | 4.9  | Green  | South-04     | 2024-05-28 |
| P0054 | Kayak Inflatable 2-Person       | 349.99 |    45 | Sports      | Intex | 4.1  | Orange | Central-02   | 2024-06-01 |
| P0055 | Atlas of World History Deluxe   |  45.00 |   678 | Books       | DK    | 4.6  | Blue   | East-03      | 2024-06-03 |
| P0056 | Bamboo Cutting Board Set 3pc    |  29.99 |   734 | Home        | Totly | 4.5  | Brown  | West-01      | 2024-06-05 |
| P0057 | RC Monster Truck 4WD 1:16       |  42.99 |   312 | Toys        | Traxx | 4.2  | Red    | North-05     | 2024-06-08 |
| P0058 | Wheelbarrow Steel 6 Cubic Ft    |  69.99 |   123 | Garden      | Marthw| 4.3  | Green  | South-04     | 2024-06-10 |
| P0059 | Webcam 4K HDR with Microphone   |  79.99 |   456 | Electronics | Logch | 4.4  | Black  | Central-02   | 2024-06-12 |
| P0060 | Cashmere Beanie Unisex          |  38.00 |   567 | Clothing    | Acne  | 4.7  | Gray   | East-03      | 2024-06-15 |
| P0061 | Dried Mango Slices Organic 500g |   7.99 |  2345 | Food        | Trdjo | 4.6  | Orange | West-01      | 2024-06-18 |
| P0062 | Boxing Gloves 14oz Professional | 64.99  |   178 | Sports      | Evrlst| 4.5  | Red    | North-05     | 2024-06-20 |
| P0063 | Childrens Illustrated Science   |  19.99 |  1234 | Books       | Usbrn | 4.8  | Yellow | South-04     | 2024-06-22 |
| P0064 | Espresso Machine Semi-Auto      | 449.99 |    67 | Home        | Brevi | 4.3  | Steel  | Central-02   | 2024-06-25 |
| P0065 | Board Game Strategy Medieval    |  39.99 |   389 | Toys        | Catan | 4.9  | Multi  | East-03      | 2024-06-28 |
| P0066 | Garden Hose 50ft Expandable     |  27.99 |   456 | Garden      | Flxzl | 4.2  | Green  | West-01      | 2024-07-01 |
| P0067 | Tablet 10-inch 128GB WiFi       | 229.99 |   234 | Electronics | Apple | 4.7  | Space  | North-05     | 2024-07-03 |
| P0068 | Denim Jacket Classic Fit        |  65.00 |   345 | Clothing    | Levis | 4.4  | Indigo | South-04     | 2024-07-05 |
| P0069 | Protein Bar Variety Pack 24ct   |  32.99 |  1567 | Food        | ClBar | 4.3  | Multi  | Central-02   | 2024-07-08 |
| P0070 | Skateboard Complete Maple Deck  |  59.99 |   201 | Sports      | Elemt | 4.5  | Maple  | East-03      | 2024-07-10 |
| P0071 | Travel Guide Europe 2024 Edn    |  21.99 |  1890 | Books       | LonPl | 4.6  | Blue   | West-01      | 2024-07-12 |
| P0072 | Weighted Blanket 15lb Queen      |  59.99 |   267 | Home        | Gravt | 4.8  | Gray   | North-05     | 2024-07-15 |
| P0073 | Art Supply Set Professional      |  54.99 |   189 | Toys        | Faber | 4.7  | Multi  | South-04     | 2024-07-18 |
| P0074 | Potting Soil Organic Mix 25L     |  12.99 |   890 | Garden      | Miracl| 4.4  | Brown  | Central-02   | 2024-07-20 |
| P0075 | External Battery Pack 26800mAh  |  45.99 |   534 | Electronics | Anker | 4.5  | Black  | East-03      | 2024-07-22 |
| P0076 | Raincoat Packable Ultralight    |  48.00 |   312 | Clothing    | Mrmot | 4.3  | Yellow | West-01      | 2024-07-25 |
| P0077 | Instant Ramen Variety Box 12pk  |  18.99 |  1678 | Food        | NisnF | 4.2  | Red    | North-05     | 2024-07-28 |
| P0078 | Rock Climbing Harness Adjust    |  74.99 |   145 | Sports      | BlkDm | 4.6  | Green  | South-04     | 2024-08-01 |
| P0079 | Manga Box Set Complete Series   |  89.99 |   678 | Books       | VizMd | 4.9  | Multi  | Central-02   | 2024-08-03 |
| P0080 | Stand Mixer 5-Quart 10-Speed    | 249.99 |    89 | Home        | KtchA | 4.7  | Red    | East-03      | 2024-08-05 |
"""
# fmt: on

_TBL_ROW_LINES = [ln for ln in _TBL_ROWS.strip().split("\n") if ln.startswith("|")]
PREFIX_TABLE = _TBL_HEADER + "\n".join(_TBL_ROW_LINES) + "\n"

# ═══════════════════════════════════════════════════════════════════════
# PREFIX 2: Encyclopedia article about fictional city (~3500 tokens)
# ═══════════════════════════════════════════════════════════════════════

PREFIX_TEXT = """\
You are a knowledgeable encyclopedia assistant. Here is an article about the fictional city of Valdoria.

## Geography

Valdoria is situated on the eastern coast of the continent of Merathia, nestled between the Crimson Mountains to the west and the Azure Sea to the east. The city spans approximately 847 square kilometers, making it the third-largest urban area in the region. The Silvermist River, originating from Lake Ethereal in the northern highlands, flows through the heart of the city before emptying into Crescent Bay. The terrain varies dramatically: the western districts feature rolling hills and dense temperate forests of oak and silverleaf, while the eastern waterfront consists of sandy beaches and limestone cliffs rising up to 120 meters above sea level. The climate is classified as maritime subtropical, with average temperatures ranging from 8 degrees Celsius in January to 28 degrees in July. Annual rainfall averages 1,240 millimeters, with the heaviest precipitation occurring during the autumn monsoon season from September to November. The city contains three major parks: Thornwall Gardens (42 hectares), Silvermist Riverside Walk (18 km long), and the Crimson Peak Nature Reserve (156 hectares) which is home to 247 bird species.

## History

Archaeological evidence suggests that the Valdoria region was first inhabited approximately 8,000 years ago by the Kethani people, who established fishing settlements along Crescent Bay. The first written records date to 1247 CE, when the merchant lord Aldric Thornwall constructed a stone fortress at the confluence of the Silvermist River and the bay. The settlement grew rapidly due to its strategic position along the Spice Route connecting the northern kingdoms to the southern trade ports. In 1412, Valdoria was officially chartered as a city by Queen Elara II of Merathia, who granted it special trading privileges that attracted merchants from across the known world. The city survived the Great Plague of 1523, which killed approximately 30 percent of its population, and the Siege of 1687, during which Admiral Corwin Blacktide blockaded the harbor for 47 days before being repelled by the city's naval defense fleet. The industrial revolution of the 1800s transformed Valdoria from a primarily mercantile center into a manufacturing powerhouse, with the Thornwall Steel Works employing over 12,000 workers at its peak in 1892. The Great Fire of 1901 destroyed much of the Old Quarter, leading to a massive rebuilding effort that gave the city its distinctive blend of classical and modern architecture. In 1947, Valdoria hosted the International Peace Conference that ended the Continental War, an event commemorated by the Peace Tower in Central Square.

## Demographics

As of the 2024 census, Valdoria has a population of 3,847,221 residents, representing an increase of 12.3 percent from the 2014 count. The city is ethnically diverse: approximately 42 percent identify as Merathian, 23 percent as Kethani, 15 percent as Eastlander, 11 percent as Southern Islander, and 9 percent as mixed or other heritage. The median age is 34.7 years. The official language is Merathian Common, though Kethani and Eastlander Trade Pidgin are widely spoken in their respective neighborhoods. The literacy rate stands at 97.8 percent. The median household income is 52,400 crowns per year, though significant disparities exist between the affluent Western Heights district (median 128,000 crowns) and the working-class Dockside Quarter (median 28,700 crowns). The Northgate district has become known as the cultural hub, home to 34 art galleries, 12 theaters, and the famous Valdoria Opera House which seats 2,800.

## Economy

Valdoria's economy is the most diversified in the Merathian region, with a gross metropolitan product of 187 billion crowns. The technology sector has become the largest employer, accounting for 28 percent of jobs, centered around the Silvermist Innovation District where major firms like CrystalTech, NovaByte, and MerathiSoft maintain their headquarters. The traditional maritime industry still contributes significantly: the Port of Valdoria handles 14.2 million metric tons of cargo annually, ranking it as the second-busiest port on the eastern seaboard. Tourism generates approximately 8.3 billion crowns per year, driven by the city's historical landmarks, the Azure Beach resort area, and the annual Festival of Lights which attracts over 2 million visitors each October. The financial district, centered on Crown Street, houses the Merathian Stock Exchange and 47 major banking institutions. Agriculture in the surrounding Valdoria Valley produces primarily wine grapes, olives, and citrus fruits, with the region's Crimson Valley Cabernet consistently ranked among the top wines globally. The unemployment rate stands at 4.2 percent, the lowest in the region.

## Education

The city is home to the University of Valdoria, founded in 1523, which enrolls approximately 45,000 students across its five campuses. The university's marine biology program is ranked first globally, and its school of engineering ranks third in the continent. The Thornwall Technical Institute, established in 1876, specializes in applied sciences and trades, serving 18,000 students. The public school system operates 312 primary schools and 87 secondary schools, employing 14,200 teachers. The Valdoria Central Library, designed by architect Mira Sunstone in 1901, holds over 4.2 million volumes and serves as the repository for the Kethani Scrolls, a collection of 847 ancient manuscripts dating to the 3rd century BCE. The city also hosts the Merathian Academy of Sciences, which awards the prestigious Thornwall Medal annually to outstanding researchers.

## Transportation

Valdoria's transportation network includes the Metro system, inaugurated in 1967, which operates 6 lines covering 142 kilometers with 98 stations, serving approximately 1.8 million riders daily. The city bus network comprises 215 routes operated by a fleet of 1,340 vehicles, of which 780 are electric. Valdoria International Airport, located 22 kilometers northeast of the city center, handles 28.4 million passengers annually and connects to 147 destinations worldwide. The High-Speed Rail terminal provides service to the capital city of Merathis (distance: 340 km, travel time: 1 hour 42 minutes) and six other major cities. The harbor ferry system operates 12 routes connecting the mainland to the offshore islands of Crescenta, Azurith, and Little Bay. The city has invested heavily in cycling infrastructure, with 487 kilometers of dedicated bike lanes and a public bike-sharing system with 15,000 bicycles across 620 stations.

## Notable Landmarks

The Crown Bridge, completed in 1934, spans 1.2 kilometers across Crescent Bay and is considered an engineering marvel of its era. The Thornwall Citadel, the original 1247 fortress, has been preserved as a museum and receives 1.4 million visitors annually. The Azure Lighthouse, standing at 67 meters tall on Point Serenity, has guided ships into the harbor since 1789 and remains operational today. The Peace Tower in Central Square, built in 1952, rises 88 meters and contains a carillon of 63 bells that chime every hour. The Silvermist Botanical Garden houses over 12,000 plant species from six continents and includes the famous Crystal Greenhouse, a 4,200 square meter glass structure designed by architect Jovan Clearwater in 1923.

## Culture and Arts

The Valdoria Philharmonic Orchestra, established in 1834, performs at the Grand Concert Hall which seats 3,200 and is renowned for its exceptional acoustics. The annual Valdoria International Film Festival, held in March, screens over 400 films from 60 countries and awards the prestigious Golden Crescent trophy. The city's culinary scene is equally vibrant, with 847 registered restaurants ranging from street food vendors in the Dockside Quarter to 14 Michelin-starred establishments in Western Heights. The Kethani Heritage Museum preserves artifacts spanning 8,000 years of indigenous history and attracts 890,000 visitors annually. Street art is officially encouraged in the Northgate district, where over 200 murals by artists from 30 countries cover building facades along the famous Color Mile. The Valdoria Jazz Festival, held outdoors in Thornwall Gardens each July, features 120 performances over 10 days and draws 180,000 attendees.

## Sports and Recreation

Valdoria is home to three professional sports teams: the Valdoria Stormhawks (football, founded 1921, playing at Crescent Stadium which seats 58,000), the Azure Bay Sharks (basketball, established 1956, based at the 14,000-seat Thornwall Arena), and the Silvermist Dolphins (swimming and water polo, competing internationally since 1968). The city hosted the Continental Games in 1988, for which it constructed the Olympic Village in the Eastern Waterfront district, now converted into a residential neighborhood housing 12,000 people. The Crimson Mountain Ski Resort, located 45 kilometers west of the city center, offers 32 marked runs and receives an average of 340,000 visitors during the winter season. Valdoria's coastline features 14 public beaches spanning 23 kilometers, with Azure Beach consistently rated among the top 10 urban beaches globally. The city maintains 127 public sports facilities including 42 swimming pools, 68 tennis courts, and the recently opened Silvermill Climbing Center, the largest indoor climbing facility in Merathia at 4,800 square meters. The annual Valdoria Marathon, first held in 1952, attracts 35,000 runners from 80 countries and follows a scenic route along the Silvermist River and Crescent Bay waterfront.

"""

# ═══════════════════════════════════════════════════════════════════════
# PREFIX 3: Python code - library management system (~3500 tokens)
# ═══════════════════════════════════════════════════════════════════════

PREFIX_CODE = '''\
You are a code analysis assistant. Below is a Python module for a library management system.

```python
"""Library Management System - Core Module
Handles book inventory, member management, and transaction tracking.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import hashlib
import json


class BookStatus(Enum):
    AVAILABLE = "available"
    CHECKED_OUT = "checked_out"
    RESERVED = "reserved"
    MAINTENANCE = "maintenance"
    LOST = "lost"


class MemberTier(Enum):
    BASIC = "basic"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"


TIER_LIMITS = {
    MemberTier.BASIC: {"max_books": 3, "max_days": 14, "late_fee": 0.50},
    MemberTier.SILVER: {"max_books": 5, "max_days": 21, "late_fee": 0.35},
    MemberTier.GOLD: {"max_books": 10, "max_days": 30, "late_fee": 0.20},
    MemberTier.PLATINUM: {"max_books": 20, "max_days": 45, "late_fee": 0.0},
}


@dataclass
class Book:
    isbn: str
    title: str
    author: str
    genre: str
    published_year: int
    page_count: int
    status: BookStatus = BookStatus.AVAILABLE
    checked_out_by: Optional[str] = None
    due_date: Optional[datetime] = None
    total_checkouts: int = 0
    average_rating: float = 0.0
    rating_count: int = 0

    def unique_id(self) -> str:
        """Generate a 12-char hex ID from ISBN + title."""
        return hashlib.md5(f"{self.isbn}-{self.title}".encode()).hexdigest()[:12]

    def is_overdue(self) -> bool:
        if self.due_date is None:
            return False
        return datetime.now() > self.due_date

    def days_overdue(self) -> int:
        if not self.is_overdue():
            return 0
        return (datetime.now() - self.due_date).days

    def add_rating(self, score: float) -> None:
        if not 1.0 <= score <= 5.0:
            raise ValueError(f"Rating must be 1.0-5.0, got {score}")
        total = self.average_rating * self.rating_count + score
        self.rating_count += 1
        self.average_rating = round(total / self.rating_count, 2)


@dataclass
class Member:
    member_id: str
    name: str
    email: str
    tier: MemberTier = MemberTier.BASIC
    join_date: datetime = field(default_factory=datetime.now)
    books_checked_out: list = field(default_factory=list)
    total_fines: float = 0.0
    is_active: bool = True

    def can_checkout(self) -> bool:
        if not self.is_active:
            return False
        limit = TIER_LIMITS[self.tier]["max_books"]
        return len(self.books_checked_out) < limit

    def checkout_slots_remaining(self) -> int:
        limit = TIER_LIMITS[self.tier]["max_books"]
        return max(0, limit - len(self.books_checked_out))

    def calculate_late_fee(self, days: int) -> float:
        rate = TIER_LIMITS[self.tier]["late_fee"]
        return round(days * rate, 2)

    def upgrade_tier(self) -> bool:
        tiers = list(MemberTier)
        idx = tiers.index(self.tier)
        if idx < len(tiers) - 1:
            self.tier = tiers[idx + 1]
            return True
        return False


class LibrarySystem:
    """Main library system managing books, members, and transactions."""

    def __init__(self, name: str, max_capacity: int = 100000):
        self.name = name
        self.max_capacity = max_capacity
        self.books: dict[str, Book] = {}
        self.members: dict[str, Member] = {}
        self.transaction_log: list[dict] = []
        self._next_transaction_id = 1000

    def add_book(self, book: Book) -> bool:
        if len(self.books) >= self.max_capacity:
            return False
        if book.isbn in self.books:
            return False
        self.books[book.isbn] = book
        self._log_transaction("add_book", book_isbn=book.isbn)
        return True

    def remove_book(self, isbn: str) -> Optional[Book]:
        book = self.books.get(isbn)
        if book is None:
            return None
        if book.status == BookStatus.CHECKED_OUT:
            return None
        del self.books[isbn]
        self._log_transaction("remove_book", book_isbn=isbn)
        return book

    def register_member(self, member: Member) -> bool:
        if member.member_id in self.members:
            return False
        self.members[member.member_id] = member
        self._log_transaction("register", member_id=member.member_id)
        return True

    def checkout_book(self, member_id: str, isbn: str) -> dict:
        member = self.members.get(member_id)
        if member is None:
            return {"success": False, "error": "Member not found"}
        if not member.can_checkout():
            return {"success": False, "error": "Checkout limit reached"}
        book = self.books.get(isbn)
        if book is None:
            return {"success": False, "error": "Book not found"}
        if book.status != BookStatus.AVAILABLE:
            return {"success": False, "error": f"Book status: {book.status.value}"}
        max_days = TIER_LIMITS[member.tier]["max_days"]
        book.status = BookStatus.CHECKED_OUT
        book.checked_out_by = member_id
        book.due_date = datetime.now() + timedelta(days=max_days)
        book.total_checkouts += 1
        member.books_checked_out.append(isbn)
        self._log_transaction("checkout", member_id=member_id,
                              book_isbn=isbn, due_days=max_days)
        return {"success": True, "due_date": book.due_date.isoformat(),
                "books_remaining": member.checkout_slots_remaining()}

    def return_book(self, member_id: str, isbn: str) -> dict:
        member = self.members.get(member_id)
        if member is None:
            return {"success": False, "error": "Member not found"}
        book = self.books.get(isbn)
        if book is None:
            return {"success": False, "error": "Book not found"}
        if book.checked_out_by != member_id:
            return {"success": False, "error": "Book not checked out by this member"}
        fine = 0.0
        if book.is_overdue():
            days = book.days_overdue()
            fine = member.calculate_late_fee(days)
            member.total_fines += fine
        book.status = BookStatus.AVAILABLE
        book.checked_out_by = None
        book.due_date = None
        member.books_checked_out.remove(isbn)
        self._log_transaction("return", member_id=member_id,
                              book_isbn=isbn, fine=fine)
        return {"success": True, "fine": fine, "total_fines": member.total_fines}

    def search_books(self, query: str, field: str = "title") -> list[Book]:
        query_lower = query.lower()
        results = []
        for book in self.books.values():
            value = getattr(book, field, "")
            if isinstance(value, str) and query_lower in value.lower():
                results.append(book)
        return sorted(results, key=lambda b: b.title)

    def get_overdue_books(self) -> list[tuple[Book, Member, int]]:
        overdue = []
        for book in self.books.values():
            if book.is_overdue() and book.checked_out_by:
                member = self.members.get(book.checked_out_by)
                if member:
                    days = (datetime.now() - book.due_date).days
                    overdue.append((book, member, days))
        return sorted(overdue, key=lambda x: x[2], reverse=True)

    def get_popular_books(self, top_n: int = 10) -> list[Book]:
        return sorted(self.books.values(),
                      key=lambda b: b.total_checkouts, reverse=True)[:top_n]

    def get_member_stats(self, member_id: str) -> Optional[dict]:
        member = self.members.get(member_id)
        if member is None:
            return None
        return {
            "name": member.name,
            "tier": member.tier.value,
            "books_out": len(member.books_checked_out),
            "slots_remaining": member.checkout_slots_remaining(),
            "total_fines": member.total_fines,
            "member_since": member.join_date.isoformat(),
        }

    def _log_transaction(self, action: str, **kwargs) -> None:
        entry = {
            "id": self._next_transaction_id,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.transaction_log.append(entry)
        self._next_transaction_id += 1
```

'''

# ═══════════════════════════════════════════════════════════════════════
# Questions: 8 prompts across 3 prefix types
# (question_suffix, expected_substring, description)
# ═══════════════════════════════════════════════════════════════════════

# Table questions (3)
TABLE_QS = [
    ("Question: what is the price of product P0027?\nAnswer: The price of P0027 is ",
     "199.99", "tbl-price-P0027"),
    ("Question: what category does product P0043 belong to?\nAnswer: P0043 belongs to ",
     "Electronics", "tbl-cat-P0043"),
    ("Question: which warehouse stores product P0053?\nAnswer: P0053 is stored in ",
     "South-04", "tbl-warehouse-P0053"),
]

# Text questions (3)
TEXT_QS = [
    ("Question: what is the population of Valdoria according to the 2024 census?\nAnswer: The population is ",
     "3,847,221", "txt-population"),
    ("Question: how many days did Admiral Corwin Blacktide blockade the harbor?\nAnswer: The blockade lasted ",
     "47", "txt-blockade-days"),
    ("Question: how many plant species does the Silvermist Botanical Garden house?\nAnswer: The garden houses ",
     "12,000", "txt-botanical-species"),
]

# Code questions (2)
CODE_QS = [
    ("Question: what is the maximum number of books a GOLD tier member can check out?\nAnswer: A GOLD member can check out up to ",
     "10", "code-gold-limit"),
    ("Question: what is the late fee per day for a BASIC tier member?\nAnswer: The late fee is ",
     "0.50", "code-basic-fee"),
]

# ─── Common VllmRunner kwargs ─────────────────────────────────────────
_COMMON_KWARGS = dict(
    model_name=MODEL,
    tensor_parallel_size=1,
    enforce_eager=True,
    gpu_memory_utilization=0.7,
    enable_prefix_caching=True,
    max_model_len=8192,
)


def _run_single_mode(mode: str, prompts: list[str]) -> list[str]:
    """Run prompts through one engine instance, return GENERATED texts only."""
    kwargs = {**_COMMON_KWARGS}
    if mode == "all":
        kwargs["mamba_cache_mode"] = "all"
    with VllmRunner(**kwargs) as runner:
        outputs = runner.generate_greedy(prompts, MAX_TOKENS)
    # generate_greedy returns (prompt+generated); strip prompt to get only generated
    results = []
    for prompt, out in zip(prompts, outputs):
        full_text = out[1]
        generated = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
        results.append(generated)
    return results


def test_precision_probe():
    """Run 8 diverse prompts through all-mode and align-mode, compare."""

    # Build all prompts with their respective prefixes
    all_qs = (
        [(PREFIX_TABLE, q, exp, desc) for q, exp, desc in TABLE_QS]
        + [(PREFIX_TEXT, q, exp, desc) for q, exp, desc in TEXT_QS]
        + [(PREFIX_CODE, q, exp, desc) for q, exp, desc in CODE_QS]
    )
    prompts = [prefix + question for prefix, question, _, _ in all_qs]

    print(f"\n{'='*70}")
    print(f"PRECISION PROBE: {len(all_qs)} diverse prompts x 2 modes")
    print(f"  Table prefix: {len(PREFIX_TABLE)} chars")
    print(f"  Text prefix:  {len(PREFIX_TEXT)} chars")
    print(f"  Code prefix:  {len(PREFIX_CODE)} chars")
    print(f"{'='*70}")

    print("\n[1/2] Running align-mode (baseline)...")
    align_texts = _run_single_mode("align", prompts)

    print("\n[2/2] Running all-mode...")
    all_texts = _run_single_mode("all", prompts)

    # Compare results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    n_answer_ok = 0
    n_eos_ok = 0
    n_match = 0

    for i, (_, question, expected, desc) in enumerate(all_qs):
        all_t = all_texts[i]
        align_t = align_texts[i]

        answer_ok = expected in all_t[:100]
        eos_ok = "Question:" not in all_t and len(all_t) < 200
        match = all_t.strip() == align_t.strip()

        if answer_ok:
            n_answer_ok += 1
        if eos_ok:
            n_eos_ok += 1
        if match:
            n_match += 1

        status = "OK" if (answer_ok and eos_ok) else "!!"
        print(f"\n  [{status}] {desc}")
        print(f"    expected: '{expected}'")
        print(f"    all:   '{all_t[:150]}' ({len(all_t)}ch)")
        print(f"    align: '{align_t[:150]}' ({len(align_t)}ch)")
        flags = []
        flags.append(f"answer={'OK' if answer_ok else 'WRONG'}")
        flags.append(f"eos={'OK' if eos_ok else 'NO-STOP'}")
        flags.append(f"match={'YES' if match else 'NO'}")
        print(f"    {' | '.join(flags)}")

    print(f"\n{'='*70}")
    print(f"SUMMARY ({len(all_qs)} prompts)")
    print(f"  Answer correct: {n_answer_ok}/{len(all_qs)}")
    print(f"  EOS clean stop: {n_eos_ok}/{len(all_qs)}")
    print(f"  all==align:     {n_match}/{len(all_qs)}")
    print(f"{'='*70}")

CREATE TABLE performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_id INTEGER REFERENCES routes(route_id),
                stop_id INTEGER REFERENCES stops(stop_id),
                service_date DATE,
                scheduled_time TEXT,
                actual_time TEXT,
                delay_minutes REAL
            );

CREATE TABLE route_stops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_id INTEGER REFERENCES routes(route_id),
                stop_id INTEGER REFERENCES stops(stop_id),
                stop_sequence INTEGER
            );

CREATE TABLE routes (
                route_id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_number TEXT,
                route_name TEXT
            );

CREATE TABLE stops (
                stop_id INTEGER PRIMARY KEY AUTOINCREMENT,
                stop_name TEXT,
                daily_passengers REAL,
                annual_passengers REAL
            );


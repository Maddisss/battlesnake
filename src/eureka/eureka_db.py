import sqlite3
from pathlib import Path
from datetime import datetime
import json

DB_PATH = Path("eureka_candidates.db")


class EurekaDB:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS reward_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation INTEGER NOT NULL,
            parent_id INTEGER,
            fitness REAL,
            rewards REAL,
            code TEXT NOT NULL,
            episode_stats TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY(parent_id) REFERENCES reward_candidates(id)
        );
        """)
        self.conn.commit()

    # ------------------------------
    # Insert new reward candidate
    # ------------------------------
    def insert_candidate(self, generation, code, parent_id=None, fitness=None, rewards=None, episode_stats=None):
        # Convert episode_stats to JSON string
        stats_json = json.dumps(episode_stats) if episode_stats else None

        cur = self.conn.execute(
            """
            INSERT INTO reward_candidates (generation, parent_id, code, fitness, rewards, episode_stats)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (generation, parent_id, code, fitness, rewards, stats_json)
        )
        self.conn.commit()
        return cur.lastrowid

    # ------------------------------
    # Update fitness and optionally episode stats
    # ------------------------------
    def update_fitness(self, candidate_id, fitness, episode_stats=None):
        stats_json = json.dumps(episode_stats) if episode_stats else None

        self.conn.execute("""
            UPDATE reward_candidates
            SET fitness = ?, episode_stats = ?, status = 'evaluated'
            WHERE id = ?
        """, (fitness, stats_json, candidate_id))
        self.conn.commit()

    # ------------------------------
    # Mark best
    # ------------------------------
    def mark_best(self, candidate_id):
        self.conn.execute("""
            UPDATE reward_candidates
            SET status = 'best'
            WHERE id = ?
        """, (candidate_id,))
        self.conn.commit()

    # ------------------------------
    # Fetch all candidates by generation
    # ------------------------------
    def fetch_by_generation(self, generation):
        cur = self.conn.execute("""
            SELECT * FROM reward_candidates
            WHERE generation = ?
            ORDER BY fitness DESC
        """, (generation,))
        return cur.fetchall()

    # ------------------------------
    # Get top performers
    # ------------------------------
    def top_candidates(self, limit=5):
        cur = self.conn.execute("""
            SELECT * FROM reward_candidates
            WHERE fitness IS NOT NULL
            ORDER BY fitness DESC
            LIMIT ?
        """, (limit,))
        return cur.fetchall()

    # ------------------------------
    # Fetch episode stats as Python objects
    # ------------------------------
    def get_episode_stats(self, candidate_id):
        cur = self.conn.execute("""
            SELECT episode_stats FROM reward_candidates WHERE id = ?
        """, (candidate_id,))
        row = cur.fetchone()
        if row and row["episode_stats"]:
            return json.loads(row["episode_stats"])
        return None

    def close(self):
        self.conn.close()

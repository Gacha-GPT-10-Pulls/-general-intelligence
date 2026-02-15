"""
ISn v4.0 SUPREME — Motor CMD Descentralizado
Operador: open_claw
Modelo de Datos Disparejos + Mercado de Prompts + Seguridad Criptografica
"""

import sys, os, math, uuid, datetime, sqlite3, json, hashlib, hmac, base64, shlex, subprocess, socket, platform, getpass, secrets, io, random

# UTF-8 en Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "isn_config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version":"4.0","identity":"open_claw","database":"isn_data.db",
            "bit_f":{"v_min":0.78,"v_max":1.67,"k_factor":0.012,"n_mid":500,
                     "max_bits":1000,"alpha_m":0.15,"beta_m":25.0,"lambda_f":0.12,"epsilon_v":0.35},
            "prompt":{"p_min":0.24,"p_max":1.24,"phi_alpha":0.22,"omega_beta":0.18,
                      "omega_gamma":3.5,"delta_floor":0.10},"export_dir":"exports"}

CFG = load_config()
V_MIN=CFG["bit_f"]["v_min"]; V_MAX=CFG["bit_f"]["v_max"]
K_FACTOR=CFG["bit_f"]["k_factor"]; N_MID=CFG["bit_f"]["n_mid"]
MAX_BITS=CFG["bit_f"]["max_bits"]; ALPHA_M=CFG["bit_f"]["alpha_m"]
BETA_M=CFG["bit_f"]["beta_m"]; LAMBDA_F=CFG["bit_f"]["lambda_f"]
EPSILON_V=CFG["bit_f"]["epsilon_v"]
P_MIN=CFG["prompt"]["p_min"]; P_MAX=CFG["prompt"]["p_max"]
PHI_ALPHA=CFG["prompt"]["phi_alpha"]; OMEGA_BETA=CFG["prompt"]["omega_beta"]
OMEGA_GAMMA=CFG["prompt"]["omega_gamma"]; DELTA_FLOOR=CFG["prompt"]["delta_floor"]
WORLD_CFG = CFG.get("world", {"need_decay_rate":3.5,"emotion_decay_rate":8.0,"event_probability":0.7,
    "interaction_impact":15.0,"mood_weights":{"needs":0.4,"emotions":0.35,"personality":0.25},"age_per_tick":1})

NODE_ID = format(uuid.getnode(), "012x")
DB_PATH = os.path.join(SCRIPT_DIR, CFG.get("database","isn_data.db"))
EXPORT_DIR = os.path.join(SCRIPT_DIR, CFG.get("export_dir","exports"))
VERSION = "4.0"; IDENTITY = CFG.get("identity","openbdf")
AUTH_PATH = os.path.join(SCRIPT_DIR, ".isn_auth")

# ═══════════════════════════════════════════════════════════════
#  MOTOR CRIPTOGRAFICO
# ═══════════════════════════════════════════════════════════════
class CryptoEngine:
    @staticmethod
    def sha512(data): return hashlib.sha512(data.encode()).hexdigest()
    @staticmethod
    def sha256(data): return hashlib.sha256(data.encode()).hexdigest()
    @staticmethod
    def hmac_sign(key, data): return hmac.new(key.encode(), data.encode(), hashlib.sha512).hexdigest()
    @staticmethod
    def hmac_verify(key, data, sig): return hmac.compare_digest(CryptoEngine.hmac_sign(key, data), sig)
    @staticmethod
    def gen_salt(): return secrets.token_hex(32)
    @staticmethod
    def derive_key(passphrase, salt):
        return hashlib.pbkdf2_hmac('sha512', passphrase.encode(), salt.encode(), 200000).hex()
    @staticmethod
    def encrypt_aes_sim(text, key):
        """Cifrado XOR multicapa con derivacion de clave (portable sin deps externas)"""
        key_bytes = hashlib.sha256(key.encode()).digest()
        data = text.encode('utf-8')
        iv = secrets.token_bytes(16)
        encrypted = bytearray(len(data))
        for i in range(len(data)):
            kb = key_bytes[(i + iv[i % 16]) % len(key_bytes)]
            encrypted[i] = data[i] ^ kb ^ iv[i % 16]
        return base64.b64encode(iv + bytes(encrypted)).decode()
    @staticmethod
    def decrypt_aes_sim(token, key):
        key_bytes = hashlib.sha256(key.encode()).digest()
        raw = base64.b64decode(token)
        iv = raw[:16]; data = raw[16:]
        decrypted = bytearray(len(data))
        for i in range(len(data)):
            kb = key_bytes[(i + iv[i % 16]) % len(key_bytes)]
            decrypted[i] = data[i] ^ kb ^ iv[i % 16]
        return bytes(decrypted).decode('utf-8')
    @staticmethod
    def gen_ihsd(node, uid, timestamp, content_hash):
        """Genera IHSD: Identificador Hash Seguro Descentralizado"""
        seg1 = CryptoEngine.sha256(node + uid)[:4].upper()
        seg2 = CryptoEngine.sha256(timestamp)[:4].upper()
        seg3 = CryptoEngine.sha256(content_hash)[:4].upper()
        combined = node + uid + timestamp + content_hash
        seg4 = CryptoEngine.sha256(combined)[:4].upper()
        return f"ISN-{seg1}-{seg2}-{seg3}-{seg4}"
    @staticmethod
    def validate_ihsd(ihsd, node, uid, timestamp, content_hash):
        expected = CryptoEngine.gen_ihsd(node, uid, timestamp, content_hash)
        return hmac.compare_digest(ihsd, expected)

CRYPTO = CryptoEngine()

# ═══════════════════════════════════════════════════════════════
#  AUTENTICACION
# ═══════════════════════════════════════════════════════════════
class AuthSystem:
    def __init__(self):
        self.authenticated = False
        self.session_token = None
        self.session_start = None
        self.failed_attempts = 0
        self.max_attempts = 5

    def is_first_run(self):
        return not os.path.exists(AUTH_PATH)

    def setup_passphrase(self, passphrase):
        if len(passphrase) < 6:
            return False, "Clave debe tener minimo 6 caracteres"
        salt = CRYPTO.gen_salt()
        derived = CRYPTO.derive_key(passphrase, salt)
        with open(AUTH_PATH, "w") as f:
            json.dump({"salt": salt, "key_hash": derived, "created": datetime.datetime.now().isoformat(),
                        "node": NODE_ID, "identity": IDENTITY}, f)
        return True, "Clave maestra configurada"

    def authenticate(self, passphrase):
        if self.failed_attempts >= self.max_attempts:
            return False, f"BLOQUEADO — {self.max_attempts} intentos fallidos"
        if not os.path.exists(AUTH_PATH):
            return False, "No hay clave configurada. Use 'setup'"
        with open(AUTH_PATH, "r") as f:
            auth = json.load(f)
        derived = CRYPTO.derive_key(passphrase, auth["salt"])
        if hmac.compare_digest(derived, auth["key_hash"]):
            self.authenticated = True
            self.session_token = secrets.token_hex(32)
            self.session_start = datetime.datetime.now()
            self.failed_attempts = 0
            return True, f"Autenticado. Token: {self.session_token[:16]}..."
        self.failed_attempts += 1
        left = self.max_attempts - self.failed_attempts
        return False, f"Clave incorrecta. {left} intentos restantes"

    def lock(self):
        self.authenticated = False
        self.session_token = None

    def require_auth(self):
        return self.authenticated

# ═══════════════════════════════════════════════════════════════
#  BASE DE DATOS
# ═══════════════════════════════════════════════════════════════
class ISNDatabase:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS records (
                uuid TEXT PRIMARY KEY, node_id TEXT, record_type TEXT,
                payload TEXT, prompt_text TEXT, hash_sha256 TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS marketplace (
                code TEXT PRIMARY KEY, ihsd TEXT UNIQUE, owner TEXT,
                model_name TEXT, description TEXT, prompt_json TEXT,
                price REAL, status TEXT DEFAULT 'available',
                created_at TEXT, signature TEXT, node_id TEXT
            );
            CREATE TABLE IF NOT EXISTS contracts (
                contract_id TEXT PRIMARY KEY, model_code TEXT,
                ihsd TEXT, seller TEXT, buyer TEXT,
                price REAL, prompt_json TEXT,
                contract_hash TEXT, signature TEXT,
                created_at TEXT, node_id TEXT,
                FOREIGN KEY (model_code) REFERENCES marketplace(code)
            );
            CREATE TABLE IF NOT EXISTS activity_log (
                log_id TEXT PRIMARY KEY,
                command TEXT, args TEXT,
                output TEXT, output_hash TEXT,
                created_at TEXT, node_id TEXT,
                session_token TEXT
            );
            CREATE TABLE IF NOT EXISTS world_inhabitants (
                name TEXT PRIMARY KEY, prompt_text TEXT,
                personality TEXT, needs TEXT, emotions TEXT,
                mood REAL DEFAULT 50.0, age INTEGER DEFAULT 0,
                alive INTEGER DEFAULT 1, history TEXT DEFAULT '[]',
                created_at TEXT, node_id TEXT
            );
            CREATE TABLE IF NOT EXISTS world_events (
                event_id TEXT PRIMARY KEY, inhabitant TEXT,
                event_type TEXT, description TEXT, impact TEXT,
                created_at TEXT, tick INTEGER,
                FOREIGN KEY (inhabitant) REFERENCES world_inhabitants(name)
            );
        """)
        self.conn.commit()

    def save(self, rtype, payload):
        uid = str(uuid.uuid4()); now = datetime.datetime.now().isoformat()
        pstr = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        h = CRYPTO.sha256(pstr)
        lines = []
        if rtype == "bit_f":
            lines = [f"[ISn Bit [f]]", f"n={payload.get('n','?')}", f"V(f)={payload.get('v_final',0):.6f}$",
                     f"C_total={payload.get('c_total',0):,.2f}$"]
        elif rtype == "prompt":
            lines = [f"[ISn Prompt]", f"Nombre:{payload.get('name','?')}", f"P(o)={payload.get('p_final',0):.6f}"]
        pt = "\n".join(lines)
        self.conn.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?)", (uid, NODE_ID, rtype, pstr, pt, h, now))
        self.conn.commit()
        return {"uuid": uid, "node_id": NODE_ID, "hash": h, "created_at": now, "prompt_text": pt}

    def get_all(self, rtype=None, limit=50):
        q, p = "SELECT * FROM records", []
        if rtype: q += " WHERE record_type=?"; p.append(rtype)
        q += " ORDER BY created_at DESC LIMIT ?"; p.append(limit)
        return [dict(r) for r in self.conn.execute(q, p).fetchall()]

    def count(self, rtype=None):
        if rtype: return self.conn.execute("SELECT COUNT(*) FROM records WHERE record_type=?", (rtype,)).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def search(self, term):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM records WHERE payload LIKE ? OR prompt_text LIKE ? LIMIT 20",
            (f"%{term}%", f"%{term}%")).fetchall()]

    def get_by_uuid(self, prefix):
        rows = self.conn.execute("SELECT * FROM records WHERE uuid LIKE ? LIMIT 1", (f"{prefix}%",)).fetchall()
        return dict(rows[0]) if rows else None

    def export_json(self, fp):
        rows = [dict(r) for r in self.conn.execute("SELECT * FROM records").fetchall()]
        with open(fp, "w", encoding="utf-8") as f: json.dump(rows, f, ensure_ascii=False, indent=2)
        return len(rows)

    def import_json(self, fp):
        with open(fp, "r", encoding="utf-8") as f: data = json.load(f)
        imp, dup = 0, 0
        for rec in data:
            try:
                pl = rec["payload"] if isinstance(rec["payload"], str) else json.dumps(rec["payload"])
                self.conn.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?)",
                    (rec["uuid"],rec["node_id"],rec["record_type"],pl,rec["prompt_text"],rec["hash_sha256"],rec["created_at"]))
                imp += 1
            except sqlite3.IntegrityError: dup += 1
        self.conn.commit()
        return imp, dup

    # Marketplace
    def create_model(self, owner, name, desc, prompt_json, price):
        code = f"MDL-{secrets.token_hex(4).upper()}"
        now = datetime.datetime.now().isoformat()
        pj = json.dumps(prompt_json, ensure_ascii=False) if isinstance(prompt_json, dict) else prompt_json
        content_hash = CRYPTO.sha256(pj)
        ihsd = CRYPTO.gen_ihsd(NODE_ID, code, now, content_hash)
        sig_data = f"{code}|{ihsd}|{owner}|{now}|{content_hash}"
        signature = CRYPTO.hmac_sign(NODE_ID, sig_data)
        self.conn.execute("INSERT INTO marketplace VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (code, ihsd, owner, name, desc, pj, price, "available", now, signature, NODE_ID))
        self.conn.commit()
        return {"code": code, "ihsd": ihsd, "owner": owner, "name": name, "price": price,
                "signature": signature[:32], "created_at": now}

    def list_models(self, status=None):
        q, p = "SELECT * FROM marketplace", []
        if status: q += " WHERE status=?"; p.append(status)
        q += " ORDER BY created_at DESC"
        return [dict(r) for r in self.conn.execute(q, p).fetchall()]

    def get_model(self, code):
        rows = self.conn.execute("SELECT * FROM marketplace WHERE code=?", (code,)).fetchall()
        return dict(rows[0]) if rows else None

    def buy_model(self, code, buyer):
        model = self.get_model(code)
        if not model: return None, "Modelo no encontrado"
        if model["status"] != "available": return None, "Modelo no disponible"
        cid = f"CTR-{secrets.token_hex(6).upper()}"
        now = datetime.datetime.now().isoformat()
        contract_data = f"{cid}|{code}|{model['ihsd']}|{model['owner']}|{buyer}|{model['price']}|{now}"
        contract_hash = CRYPTO.sha256(contract_data)
        signature = CRYPTO.hmac_sign(NODE_ID, contract_data)
        self.conn.execute("INSERT INTO contracts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (cid, code, model["ihsd"], model["owner"], buyer, model["price"],
             model["prompt_json"], contract_hash, signature, now, NODE_ID))
        self.conn.execute("UPDATE marketplace SET status='sold' WHERE code=?", (code,))
        self.conn.commit()
        return {"contract_id": cid, "code": code, "ihsd": model["ihsd"], "seller": model["owner"],
                "buyer": buyer, "price": model["price"], "hash": contract_hash,
                "signature": signature[:32], "created_at": now}, None

    def list_contracts(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM contracts ORDER BY created_at DESC").fetchall()]

    def get_contract(self, cid):
        rows = self.conn.execute("SELECT * FROM contracts WHERE contract_id LIKE ?", (f"{cid}%",)).fetchall()
        return dict(rows[0]) if rows else None

    def nodes(self):
        return [r[0] for r in self.conn.execute("SELECT DISTINCT node_id FROM records").fetchall()]

    def close(self): self.conn.close()

    def log_activity(self, cmd, args, output, session_token=None):
        """Guardar actividad en la DB"""
        try:
            log_id = f"LOG-{secrets.token_hex(6).upper()}"
            now = datetime.datetime.now().isoformat()
            args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, list) else str(args)
            out_hash = CRYPTO.sha256(output) if output else ""
            self.conn.execute("INSERT INTO activity_log VALUES (?,?,?,?,?,?,?,?)",
                (log_id, cmd, args_str, output[:2000] if output else "", out_hash, now, NODE_ID, session_token or ""))
            self.conn.commit()
        except: pass

    # ── WORLD DB METHODS ──
    def save_inhabitant(self, name, prompt_text, personality, needs, emotions, mood=50.0):
        now = datetime.datetime.now().isoformat()
        self.conn.execute("INSERT OR REPLACE INTO world_inhabitants VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (name, prompt_text, json.dumps(personality), json.dumps(needs),
             json.dumps(emotions), mood, 0, 1, '[]', now, NODE_ID))
        self.conn.commit()

    def get_inhabitant(self, name):
        r = self.conn.execute("SELECT * FROM world_inhabitants WHERE name=?", (name,)).fetchone()
        return dict(r) if r else None

    def list_inhabitants(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM world_inhabitants WHERE alive=1 ORDER BY created_at DESC").fetchall()]

    def update_inhabitant(self, name, **kwargs):
        sets, vals = [], []
        for k, v in kwargs.items():
            if k in ('personality','needs','emotions','history'):
                sets.append(f"{k}=?"); vals.append(json.dumps(v))
            else:
                sets.append(f"{k}=?"); vals.append(v)
        vals.append(name)
        self.conn.execute(f"UPDATE world_inhabitants SET {','.join(sets)} WHERE name=?", vals)
        self.conn.commit()

    def save_event(self, inhabitant, event_type, description, impact, tick):
        eid = f"EVT-{secrets.token_hex(4).upper()}"
        now = datetime.datetime.now().isoformat()
        self.conn.execute("INSERT INTO world_events VALUES (?,?,?,?,?,?,?)",
            (eid, inhabitant, event_type, description, json.dumps(impact), now, tick))
        self.conn.commit()
        return eid

    def get_events(self, inhabitant=None, limit=20):
        if inhabitant:
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM world_events WHERE inhabitant=? ORDER BY created_at DESC LIMIT ?",
                (inhabitant, limit)).fetchall()]
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM world_events ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()]

# ═══════════════════════════════════════════════════════════════
#  MUNDO DE PROMPTS — MOTOR PSICOLOGICO
#  Cada prompt es una persona con psicologia real
# ═══════════════════════════════════════════════════════════════
_POSITIVE = {"amor","feliz","alegr","bien","bueno","genial","hermoso","lindo","paz","calma",
    "amable","dulce","bondad","luz","sol","sonri","maravill","excelent","increíble","fantástic",
    "love","happy","good","great","beautiful","kind","peace","joy","wonderful","nice"}
_NEGATIVE = {"odio","mal","triste","oscur","dolor","miedo","furia","rabia","muerte","solo",
    "sufr","angust","terror","cruel","violenc","destru","guerra","pelea","castigo","toxic",
    "hate","bad","sad","dark","pain","fear","anger","death","alone","suffer"}
_FORMAL = {"mediante","sin embargo","no obstante","por consiguiente","therefore","however","furthermore"}
_CREATIVE = {"imagin","sueñ","creat","arte","musa","inspir","inventa","fantasia","magia",
    "dream","imagine","create","art","magic","inspire","wonder","fantasy","invent"}

def _derive_personality(text):
    t = text.lower(); words = t.split(); wc = max(len(words), 1)
    o_len = min(len(t)/200.0,1.0)*30; o_cr = sum(1 for w in words if any(c in w for c in _CREATIVE))/wc*200
    openness = min(max(35+o_len+o_cr+t.count('?')*8, 5), 95)
    c_fm = sum(1 for f in _FORMAL if f in t)*12; c_p = (t.count('.')+t.count(','))/wc*60
    conscientiousness = min(max(40+c_fm+c_p, 5), 95)
    e_ex = t.count('!')*10; e_em = sum(1 for c in t if ord(c)>127)*3
    extraversion = min(max(35+e_ex+e_em, 5), 95)
    pos = sum(1 for w in words if any(p in w for p in _POSITIVE))
    neg = sum(1 for w in words if any(n in w for n in _NEGATIVE))
    agreeableness = min(max(50+(pos-neg)/max(wc,1)*150, 5), 95)
    neuroticism = min(max(30+neg/max(wc,1)*200+(t.count('!')+t.count('?')+t.count('...'))*4, 5), 95)
    return {"O":round(openness,1),"C":round(conscientiousness,1),"E":round(extraversion,1),
            "A":round(agreeableness,1),"N":round(neuroticism,1)}

def _initial_needs():
    return {"fisiologicas":85+random.randint(-10,10),"seguridad":80+random.randint(-10,10),
            "sociales":50+random.randint(-15,15),"estima":40+random.randint(-10,10),
            "autorrealizacion":20+random.randint(-10,10)}

def _initial_emotions():
    return {"alegria":60+random.randint(-10,20),"tristeza":5+random.randint(0,10),
            "miedo":10+random.randint(0,10),"ira":3+random.randint(0,5),
            "sorpresa":30+random.randint(-10,20),"asco":2+random.randint(0,5)}

WORLD_EVENTS = {
    "descubrimiento":{"desc":"descubrio algo fascinante","emotions":{"alegria":15,"sorpresa":25},"needs":{"autorrealizacion":10,"estima":5},"pp":"O"},
    "amistad":{"desc":"hizo una nueva conexion significativa","emotions":{"alegria":20},"needs":{"sociales":20,"seguridad":5},"pp":"A"},
    "logro":{"desc":"alcanzo una meta importante","emotions":{"alegria":25,"sorpresa":10},"needs":{"estima":20,"autorrealizacion":15},"pp":"C"},
    "conflicto":{"desc":"tuvo un conflicto con alguien","emotions":{"ira":20,"tristeza":15},"needs":{"sociales":-15,"seguridad":-10},"pp":"A"},
    "fracaso":{"desc":"experimento un fracaso importante","emotions":{"tristeza":25,"asco":10},"needs":{"estima":-20,"autorrealizacion":-15},"pp":"C"},
    "enfermedad":{"desc":"se enfermo","emotions":{"miedo":20,"tristeza":15},"needs":{"fisiologicas":-25,"seguridad":-20},"pp":"O"},
    "ayuda":{"desc":"ayudo a alguien","emotions":{"alegria":15,"sorpresa":5},"needs":{"sociales":15,"estima":10},"pp":"A"},
    "inspiracion":{"desc":"fue inspirado por algo","emotions":{"alegria":20,"sorpresa":15},"needs":{"autorrealizacion":15,"estima":10},"pp":"O"},
    "traicion":{"desc":"fue traicionado","emotions":{"ira":25,"tristeza":20,"miedo":10},"needs":{"seguridad":-25,"sociales":-20},"pp":"A"},
    "exito":{"desc":"tuvo un gran exito","emotions":{"alegria":30,"sorpresa":10},"needs":{"estima":25,"autorrealizacion":20},"pp":"C"}
}

class WorldSimulator:
    def __init__(self, database):
        self.db = database
        self.tick_count = 0
        self.running = False
        
    def create_inhabitant(self, name, prompt_text):
        """Crear un nuevo habitante en el mundo con psicología basada en su texto"""
        personality = _derive_personality(prompt_text)
        needs = _initial_needs()
        emotions = _initial_emotions()
        
        self.db.save_inhabitant(name, prompt_text, personality, needs, emotions)
        print(f"[MUNDO] Habitante '{name}' creado con psicología inicial.")
        return self.db.get_inhabitant(name)
    
    def update_needs(self, needs):
        """Actualizar las necesidades según el modelo de Maslow con decaimiento"""
        updated = {}
        for need, value in needs.items():
            # Aplicar decaimiento natural
            decayed_value = max(0, value - WORLD_CFG["need_decay_rate"])
            updated[need] = min(100, decayed_value)
        return updated
    
    def update_emotions(self, emotions):
        """Actualizar emociones con decaimiento"""
        updated = {}
        for emo, value in emotions.items():
            decayed_value = max(0, value - WORLD_CFG["emotion_decay_rate"])
            updated[emo] = decayed_value
        return updated
    
    def calculate_mood(self, needs, emotions, personality):
        """Calcular estado de ánimo basado en necesidades, emociones y personalidad"""
        # Calcular promedio ponderado de necesidades
        need_avg = sum(needs.values()) / len(needs) if needs else 50
        
        # Calcular emoción dominante (alegría vs tristeza/miedo)
        positive_emotions = emotions.get("alegria", 0) + emotions.get("sorpresa", 0)
        negative_emotions = (emotions.get("tristeza", 0) + emotions.get("miedo", 0) + 
                            emotions.get("ira", 0) + emotions.get("asco", 0))
        emotion_balance = positive_emotions - negative_emotions
        
        # Ponderaciones del estado de ánimo
        mood_weights = WORLD_CFG["mood_weights"]
        
        # Calcular estado de ánimo final
        mood = (need_avg * mood_weights["needs"] + 
                emotion_balance * mood_weights["emotions"] +
                (personality["A"] + personality["O"]) * mood_weights["personality"])
        
        return max(0, min(100, mood))
    
    def generate_event(self, inhabitant_data):
        """Generar un evento aleatorio para un habitante"""
        if random.random() > WORLD_CFG["event_probability"]:
            return None
            
        # Seleccionar un evento basado en rasgos de personalidad
        personality = json.loads(inhabitant_data["personality"])
        needs = json.loads(inhabitant_data["needs"])
        emotions = json.loads(inhabitant_data["emotions"])
        
        # Preferencias de eventos según personalidad
        possible_events = []
        for event_type, event_info in WORLD_EVENTS.items():
            pp_trait = event_info.get("pp", "O")  # Rasgo de personalidad asociado
            trait_value = personality.get(pp_trait, 50)
            
            # Los eventos que coinciden con la personalidad tienen mayor probabilidad
            weight = 0.5 + (trait_value / 200.0)  # Base 0.5 + hasta 0.25 de personalidad
            if random.random() < weight:
                possible_events.append(event_type)
        
        if not possible_events:
            return random.choice(list(WORLD_EVENTS.keys()))
            
        return random.choice(possible_events)
    
    def apply_event(self, inhabitant_name, event_type):
        """Aplicar un evento a un habitante y actualizar su estado"""
        inhabitant = self.db.get_inhabitant(inhabitant_name)
        if not inhabitant:
            return None
            
        event_info = WORLD_EVENTS[event_type]
        
        # Cargar datos actuales
        personality = json.loads(inhabitant["personality"])
        needs = json.loads(inhabitant["needs"])
        emotions = json.loads(inhabitant["emotions"])
        history = json.loads(inhabitant["history"])
        
        # Aplicar impactos del evento
        for emo, impact in event_info["emotions"].items():
            emotions[emo] = max(0, min(100, emotions.get(emo, 0) + impact))
            
        for need, impact in event_info["needs"].items():
            needs[need] = max(0, min(100, needs.get(need, 50) + impact))
        
        # Actualizar historia
        event_record = {
            "tick": self.tick_count,
            "event_type": event_type,
            "description": event_info["desc"],
            "impact": {"emotions": event_info["emotions"], "needs": event_info["needs"]}
        }
        history.append(event_record)
        
        # Guardar cambios en la base de datos
        mood = self.calculate_mood(needs, emotions, personality)
        age = inhabitant["age"] + WORLD_CFG["age_per_tick"]
        
        self.db.update_inhabitant(
            inhabitant_name,
            needs=needs,
            emotions=emotions,
            mood=mood,
            age=age,
            history=history
        )
        
        # Registrar el evento
        self.db.save_event(
            inhabitant_name,
            event_type,
            event_info["desc"],
            {"emotions": event_info["emotions"], "needs": event_info["needs"]},
            self.tick_count
        )
        
        return {
            "inhabitant": inhabitant_name,
            "event_type": event_type,
            "description": event_info["desc"],
            "mood": mood
        }
    
    def simulate_tick(self):
        """Simular un ciclo del mundo (tick)"""
        inhabitants = self.db.list_inhabitants()
        events_occurred = []
        
        for inhabitant in inhabitants:
            # Actualizar necesidades y emociones con decaimiento
            needs = json.loads(inhabitant["needs"])
            emotions = json.loads(inhabitant["emotions"])
            
            updated_needs = self.update_needs(needs)
            updated_emotions = self.update_emotions(emotions)
            
            # Intentar generar un evento
            event_type = self.generate_event(inhabitant)
            if event_type:
                event_result = self.apply_event(inhabitant["name"], event_type)
                if event_result:
                    events_occurred.append(event_result)
            else:
                # Solo actualizar con decaimiento
                personality = json.loads(inhabitant["personality"])
                mood = self.calculate_mood(updated_needs, updated_emotions, personality)
                
                self.db.update_inhabitant(
                    inhabitant["name"],
                    needs=updated_needs,
                    emotions=updated_emotions,
                    mood=mood
                )
        
        self.tick_count += 1
        return events_occurred
    
    def start_simulation(self):
        """Iniciar la simulación del mundo"""
        self.running = True
        print(f"[MUNDO] Simulación iniciada. Tick: {self.tick_count}")
        
    def stop_simulation(self):
        """Detener la simulación del mundo"""
        self.running = False
        print(f"[MUNDO] Simulación detenida. Total ticks: {self.tick_count}")

# ═══════════════════════════════════════════════════════════════
#  INTERFAZ GRAFICA CON TKINTER
# ═══════════════════════════════════════════════════════════════
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
    import threading
    import time
    
    class WorldGUI:
        def __init__(self):
            self.root = tk.Tk()
            self.root.title(f"ISn v4.0 SUPREME — Mundo de Prompts")
            self.root.geometry("1200x800")
            self.root.configure(bg="#0a0a0a")
            
            # Sistema de autenticación y base de datos
            self.auth = AuthSystem()
            self.db = ISNDatabase()
            self.world = WorldSimulator(self.db)
            
            # Variables de estado
            self.simulation_running = False
            self.simulation_thread = None
            
            self.setup_gui()
            
        def setup_gui(self):
            # Estilo
            style = ttk.Style()
            style.theme_use("clam")
            style.configure("TFrame", background="#0a0a0a")
            style.configure("TLabel", background="#0a0a0a", foreground="#00ff00")
            style.configure("TButton", background="#1a1a1a", foreground="#00ff00", borderwidth=1)
            style.map("TButton", background=[('active', '#2a2a2a')])
            style.configure("Treeview", background="#1a1a1a", foreground="#00ff00", fieldbackground="#1a1a1a")
            style.configure("Treeview.Heading", background="#2a2a2a", foreground="#00ff00")
            
            # Frame principal
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Panel de autenticación
            auth_frame = ttk.LabelFrame(main_frame, text="Autenticación", padding=(10, 5))
            auth_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(auth_frame, text="Contraseña:").grid(row=0, column=0, padx=(0, 5))
            self.password_entry = ttk.Entry(auth_frame, show="*", width=20)
            self.password_entry.grid(row=0, column=1, padx=(0, 5))
            self.password_entry.bind("<Return>", lambda e: self.authenticate())
            
            self.auth_button = ttk.Button(auth_frame, text="Autenticar", command=self.authenticate)
            self.auth_button.grid(row=0, column=2, padx=(5, 0))
            
            self.status_label = ttk.Label(auth_frame, text="Estado: No autenticado", foreground="#ff6666")
            self.status_label.grid(row=0, column=3, padx=(20, 0))
            
            # Panel principal dividido
            paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
            paned_window.pack(fill=tk.BOTH, expand=True)
            
            # Panel izquierdo - Control del mundo
            left_panel = ttk.Frame(paned_window)
            paned_window.add(left_panel, weight=1)
            
            # Control de simulación
            sim_control_frame = ttk.LabelFrame(left_panel, text="Control de Simulación", padding=(10, 5))
            sim_control_frame.pack(fill=tk.X, pady=(0, 10))
            
            self.start_stop_button = ttk.Button(sim_control_frame, text="Iniciar Simulación", 
                                              command=self.toggle_simulation)
            self.start_stop_button.pack(side=tk.LEFT, padx=(0, 10))
            
            ttk.Button(sim_control_frame, text="Próximo Tick", 
                      command=self.single_tick).pack(side=tk.LEFT, padx=(0, 10))
            
            self.tick_label = ttk.Label(sim_control_frame, text="Ticks: 0")
            self.tick_label.pack(side=tk.LEFT)
            
            # Crear habitante
            create_frame = ttk.LabelFrame(left_panel, text="Crear Habitante", padding=(10, 5))
            create_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(create_frame, text="Nombre:").grid(row=0, column=0, sticky=tk.W)
            self.name_entry = ttk.Entry(create_frame, width=20)
            self.name_entry.grid(row=0, column=1, padx=(5, 10))
            
            ttk.Label(create_frame, text="Prompt:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
            self.prompt_text = scrolledtext.ScrolledText(create_frame, height=6, width=40)
            self.prompt_text.grid(row=1, column=1, columnspan=2, padx=(5, 0), pady=(5, 0), sticky=tk.EW)
            
            ttk.Button(create_frame, text="Crear Habitante", 
                      command=self.create_inhabitant).grid(row=2, column=1, pady=(10, 0))
            
            # Lista de habitantes
            inhabitant_frame = ttk.LabelFrame(left_panel, text="Habitantes", padding=(10, 5))
            inhabitant_frame.pack(fill=tk.BOTH, expand=True)
            
            columns = ("Nombre", "Edad", "Estado de Ánimo", "Vivo")
            self.inhabitant_tree = ttk.Treeview(inhabitant_frame, columns=columns, show="headings", height=10)
            
            for col in columns:
                self.inhabitant_tree.heading(col, text=col)
                self.inhabitant_tree.column(col, width=100)
            
            scrollbar = ttk.Scrollbar(inhabitant_frame, orient=tk.VERTICAL, command=self.inhabitant_tree.yview)
            self.inhabitant_tree.configure(yscroll=scrollbar.set)
            
            self.inhabitant_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            self.inhabitant_tree.bind("<Double-1>", self.show_inhabitant_details)
            
            # Panel derecho - Consola y detalles
            right_panel = ttk.Frame(paned_window)
            paned_window.add(right_panel, weight=2)
            
            # Consola de salida
            console_frame = ttk.LabelFrame(right_panel, text="Consola", padding=(10, 5))
            console_frame.pack(fill=tk.BOTH, expand=True)
            
            self.console_output = scrolledtext.ScrolledText(console_frame, height=15, bg="#1a1a1a", fg="#00ff00")
            self.console_output.pack(fill=tk.BOTH, expand=True)
            
            # Botones de utilidad
            button_frame = ttk.Frame(right_panel)
            button_frame.pack(fill=tk.X, pady=(10, 0))
            
            ttk.Button(button_frame, text="Limpiar Consola", 
                      command=self.clear_console).pack(side=tk.LEFT, padx=(0, 10))
            
            ttk.Button(button_frame, text="Exportar Datos", 
                      command=self.export_data).pack(side=tk.LEFT, padx=(0, 10))
            
            # Inicializar lista de habitantes
            self.refresh_inhabitant_list()
            
        def authenticate(self):
            password = self.password_entry.get()
            success, message = self.auth.authenticate(password)
            
            if success:
                self.status_label.config(text="Estado: Autenticado", foreground="#66ff66")
                self.auth_button.config(text="Cerrar Sesión")
                self.auth_button.config(command=self.logout)
            else:
                self.status_label.config(text=f"Estado: {message}", foreground="#ff6666")
                messagebox.showerror("Error de Autenticación", message)
        
        def logout(self):
            self.auth.lock()
            self.status_label.config(text="Estado: No autenticado", foreground="#ff6666")
            self.auth_button.config(text="Autenticar")
            self.auth_button.config(command=self.authenticate)
            self.password_entry.delete(0, tk.END)
        
        def toggle_simulation(self):
            if not self.auth.require_auth():
                messagebox.showwarning("Advertencia", "Debe autenticarse primero")
                return
                
            if not self.simulation_running:
                self.start_simulation()
            else:
                self.stop_simulation()
        
        def start_simulation(self):
            if self.simulation_running:
                return
                
            self.simulation_running = True
            self.start_stop_button.config(text="Detener Simulación")
            self.world.start_simulation()
            
            def run_simulation():
                while self.simulation_running:
                    if self.simulation_running:
                        events = self.world.simulate_tick()
                        self.tick_label.config(text=f"Ticks: {self.world.tick_count}")
                        
                        if events:
                            for event in events:
                                self.log_message(f"[TICK {self.world.tick_count}] {event['inhabitant']}: {event['description']} (ánimo: {event['mood']:.1f})")
                        
                        time.sleep(1)  # Actualizar cada segundo
                        
            self.simulation_thread = threading.Thread(target=run_simulation, daemon=True)
            self.simulation_thread.start()
            
            self.log_message("[MUNDO] Simulación iniciada")
        
        def stop_simulation(self):
            if not self.simulation_running:
                return
                
            self.simulation_running = False
            self.start_stop_button.config(text="Iniciar Simulación")
            self.world.stop_simulation()
            self.log_message("[MUNDO] Simulación detenida")
        
        def single_tick(self):
            if not self.auth.require_auth():
                messagebox.showwarning("Advertencia", "Debe autenticarse primero")
                return
                
            events = self.world.simulate_tick()
            self.tick_label.config(text=f"Ticks: {self.world.tick_count}")
            
            if events:
                for event in events:
                    self.log_message(f"[TICK {self.world.tick_count}] {event['inhabitant']}: {event['description']} (ánimo: {event['mood']:.1f})")
            else:
                self.log_message(f"[TICK {self.world.tick_count}] No ocurrieron eventos")
        
        def create_inhabitant(self):
            if not self.auth.require_auth():
                messagebox.showwarning("Advertencia", "Debe autenticarse primero")
                return
                
            name = self.name_entry.get().strip()
            prompt = self.prompt_text.get("1.0", tk.END).strip()
            
            if not name or not prompt:
                messagebox.showwarning("Advertencia", "Debe ingresar nombre y prompt")
                return
                
            try:
                inhabitant = self.world.create_inhabitant(name, prompt)
                if inhabitant:
                    self.log_message(f"[MUNDO] Habitante '{name}' creado exitosamente")
                    self.refresh_inhabitant_list()
                    self.name_entry.delete(0, tk.END)
                    self.prompt_text.delete("1.0", tk.END)
                else:
                    messagebox.showerror("Error", "No se pudo crear el habitante")
            except Exception as e:
                messagebox.showerror("Error", f"Error al crear habitante: {str(e)}")
        
        def refresh_inhabitant_list(self):
            # Limpiar lista actual
            for item in self.inhabitant_tree.get_children():
                self.inhabitant_tree.delete(item)
            
            # Obtener habitantes de la base de datos
            inhabitants = self.db.list_inhabitants()
            
            for inhabitant in inhabitants:
                name = inhabitant["name"]
                age = inhabitant["age"]
                mood = round(inhabitant["mood"], 1)
                alive = "Sí" if inhabitant["alive"] else "No"
                
                self.inhabitant_tree.insert("", "end", values=(name, age, mood, alive))
        
        def show_inhabitant_details(self, event):
            selection = self.inhabitant_tree.selection()
            if not selection:
                return
                
            item = self.inhabitant_tree.item(selection[0])
            name = item["values"][0]
            
            inhabitant = self.db.get_inhabitant(name)
            if not inhabitant:
                return
                
            # Crear ventana de detalles
            details_window = tk.Toplevel(self.root)
            details_window.title(f"Detalles de {name}")
            details_window.geometry("600x500")
            details_window.configure(bg="#0a0a0a")
            
            # Notebook para pestañas
            notebook = ttk.Notebook(details_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Pestaña de información general
            general_frame = ttk.Frame(notebook)
            notebook.add(general_frame, text="General")
            
            info_text = scrolledtext.ScrolledText(general_frame, bg="#1a1a1a", fg="#00ff00")
            info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            info_content = f"""Nombre: {inhabitant['name']}
Edad: {inhabitant['age']}
Estado de Ánimo: {inhabitant['mood']:.1f}
Fecha de Creación: {inhabitant['created_at']}
Vivo: {'Sí' if inhabitant['alive'] else 'No'}

Prompt Original:
{inhabitant['prompt_text']}

Personalidad (OCEAN):
{json.dumps(json.loads(inhabitant['personality']), indent=2)}
"""
            info_text.insert(tk.END, info_content)
            info_text.config(state=tk.DISABLED)
            
            # Pestaña de necesidades
            needs_frame = ttk.Frame(notebook)
            notebook.add(needs_frame, text="Necesidades")
            
            needs_text = scrolledtext.ScrolledText(needs_frame, bg="#1a1a1a", fg="#00ff00")
            needs_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            needs_content = f"Necesidades Actuales:\n{json.dumps(json.loads(inhabitant['needs']), indent=2)}"
            needs_text.insert(tk.END, needs_content)
            needs_text.config(state=tk.DISABLED)
            
            # Pestaña de emociones
            emotions_frame = ttk.Frame(notebook)
            notebook.add(emotions_frame, text="Emociones")
            
            emotions_text = scrolledtext.ScrolledText(emotions_frame, bg="#1a1a1a", fg="#00ff00")
            emotions_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            emotions_content = f"Emociones Actuales:\n{json.dumps(json.loads(inhabitant['emotions']), indent=2)}"
            emotions_text.insert(tk.END, emotions_content)
            emotions_text.config(state=tk.DISABLED)
            
            # Pestaña de eventos
            events_frame = ttk.Frame(notebook)
            notebook.add(events_frame, text="Historial de Eventos")
            
            events_text = scrolledtext.ScrolledText(events_frame, bg="#1a1a1a", fg="#00ff00")
            events_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            history = json.loads(inhabitant['history'])
            events_content = f"Historial de {len(history)} eventos:\n"
            for event in history[-10:]:  # Mostrar últimos 10 eventos
                events_content += f"\nTick {event['tick']}: {event['event_type']} - {event['description']}\n"
                events_content += f"  Impacto: {event['impact']}\n"
            
            events_text.insert(tk.END, events_content)
            events_text.config(state=tk.DISABLED)
        
        def log_message(self, message):
            self.console_output.insert(tk.END, message + "\n")
            self.console_output.see(tk.END)
        
        def clear_console(self):
            self.console_output.delete("1.0", tk.END)
        
        def export_data(self):
            if not self.auth.require_auth():
                messagebox.showwarning("Advertencia", "Debe autenticarse primero")
                return
                
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                try:
                    count = self.db.export_json(filename)
                    self.log_message(f"[EXPORT] Exportados {count} registros a {filename}")
                    messagebox.showinfo("Éxito", f"Exportados {count} registros")
                except Exception as e:
                    messagebox.showerror("Error", f"Error al exportar: {str(e)}")
        
        def run(self):
            self.root.mainloop()
    
    # Ejecutar la interfaz gráfica
    if __name__ == "__main__":
        app = WorldGUI()
        app.run()
        
except ImportError:
    # Si no está disponible tkinter, ejecutar modo consola
    print("Tkinter no disponible, ejecutando en modo consola...")
    
    def run_console_mode():
        auth = AuthSystem()
        db = ISNDatabase()
        world = WorldSimulator(db)
        
        print("ISn v4.0 SUPREME — Modo Consola")
        print("="*50)
        
        if auth.is_first_run():
            print("Primera ejecución - Configurando clave maestra")
            pwd = input("Ingrese clave maestra: ")
            success, msg = auth.setup_passphrase(pwd)
            if not success:
                print(f"Error: {msg}")
                return
            print("Clave configurada correctamente")
        
        # Autenticar
        while not auth.authenticated:
            pwd = input("Ingrese contraseña: ")
            success, msg = auth.authenticate(pwd)
            print(msg)
            if not success:
                continue
        
        print("\nComandos disponibles:")
        print("1. crear_habitante <nombre> <prompt>")
        print("2. listar_habitantes")
        print("3. iniciar_simulacion")
        print("4. siguiente_tick")
        print("5. salir")
        
        world.start_simulation()
        
        while True:
            try:
                cmd = input("\n> ").strip().split()
                if not cmd:
                    continue
                    
                if cmd[0] == "crear_habitante" and len(cmd) >= 3:
                    name = cmd[1]
                    prompt = " ".join(cmd[2:])
                    inhabitant = world.create_inhabitant(name, prompt)
                    if inhabitant:
                        print(f"Habitante '{name}' creado exitosamente")
                    else:
                        print("Error al crear habitante")
                        
                elif cmd[0] == "listar_habitantes":
                    inhabitants = db.list_inhabitants()
                    print(f"\nHabitantes ({len(inhabitants)}):")
                    for inh in inhabitants:
                        print(f"- {inh['name']} | Edad: {inh['age']} | Ánimo: {inh['mood']:.1f}")
                        
                elif cmd[0] == "iniciar_simulacion":
                    world.start_simulation()
                    print("Simulación iniciada")
                    
                elif cmd[0] == "siguiente_tick":
                    events = world.simulate_tick()
                    print(f"Tick {world.tick_count} completado")
                    for event in events:
                        print(f"- {event['inhabitant']}: {event['description']} (ánimo: {event['mood']:.1f})")
                        
                elif cmd[0] == "salir":
                    break
                        
                else:
                    print("Comando no reconocido")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        world.stop_simulation()
        db.close()
    
    if __name__ == "__main__":
        run_console_mode()
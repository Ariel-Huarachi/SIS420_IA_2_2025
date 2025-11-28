import numpy as np

# ===========================================================
#   CLASE BASE AGENTE (PERSEGUIDOR / EVASOR)
# ===========================================================

class Agent:
    """Clase base para Perseguidor y Evasor en entorno continuo."""

    def __init__(self, x, y, radius=0.3, speed=1.0):
        self.x = float(x)
        self.y = float(y)
        self.radius = radius
        self.speed = speed
        self.base_speed = speed
        self.history = [(self.x, self.y)]
        
        # 🔥 NUEVO: Detección de atasco
        self.intentos_fallidos = 0
        self.ultima_direccion = None

    def get_position(self):
        return (self.x, self.y)

    def reset_history(self):
        self.history = [(self.x, self.y)]
        self.intentos_fallidos = 0

    def restore_speed(self):
        """Restaurar velocidad base."""
        self.speed = self.base_speed

    # ===========================================================
    #   🔥 SISTEMA ANTI-ATASCOS MEJORADO
    # ===========================================================
    def move_angle(self, angle_deg, maze):
        """
        Movimiento continuo mejorado CON ANTI-ATASCOS AGRESIVO
        """
        pos_inicial = (self.x, self.y)
        
        rad = np.radians(angle_deg)
        dx = np.cos(rad) * self.speed
        dy = np.sin(rad) * self.speed

        new_x = self.x + dx
        new_y = self.y + dy

        # --- INTENTO 1: Movimiento normal ---
        if not maze.check_collision(new_x, new_y, self.radius):
            self.x = new_x
            self.y = new_y
            self.history.append((self.x, self.y))
            self.intentos_fallidos = 0  # ← Reset contador
            self.ultima_direccion = angle_deg
            return True

        # --- HAY COLISIÓN - INTENTOS ALTERNATIVOS ---
        
        # INTENTO 2: Solo eje X
        if not maze.check_collision(self.x + dx, self.y, self.radius):
            self.x += dx
            self.history.append((self.x, self.y))
            self.intentos_fallidos = 0
            return True

        # INTENTO 3: Solo eje Y
        if not maze.check_collision(self.x, self.y + dy, self.radius):
            self.y += dy
            self.history.append((self.x, self.y))
            self.intentos_fallidos = 0
            return True

        # INTENTO 4: Diagonal reducida (50%)
        if not maze.check_collision(self.x + dx*0.5, self.y + dy*0.5, self.radius):
            self.x += dx * 0.5
            self.y += dy * 0.5
            self.history.append((self.x, self.y))
            self.intentos_fallidos = 0
            return True

        # INTENTO 5: Probar ángulos cercanos (±45°)
        for offset in [45, -45, 90, -90]:
            test_angle = angle_deg + offset
            test_rad = np.radians(test_angle)
            test_dx = np.cos(test_rad) * self.speed
            test_dy = np.sin(test_rad) * self.speed
            
            if not maze.check_collision(self.x + test_dx, self.y + test_dy, self.radius):
                self.x += test_dx
                self.y += test_dy
                self.history.append((self.x, self.y))
                self.intentos_fallidos = 0
                return True

        # 🔥 NUEVO: INTENTO 6 - Sistema de empuje agresivo
        self.intentos_fallidos += 1
        
        if self.intentos_fallidos > 3:  # Está muy atascado
            # Empuje grande en direcciones cardinales
            empuje = 0.2  # ← Aumentado de 0.05
            
            direcciones = [
                (empuje, 0), (-empuje, 0),
                (0, empuje), (0, -empuje),
                (empuje, empuje), (-empuje, -empuje),
                (empuje, -empuje), (-empuje, empuje)
            ]
            
            for push_x, push_y in direcciones:
                if not maze.check_collision(self.x + push_x, self.y + push_y, self.radius):
                    self.x += push_x
                    self.y += push_y
                    self.history.append((self.x, self.y))
                    self.intentos_fallidos = 0
                    return True
        
        # 🔥 NUEVO: INTENTO 7 - Teletransporte mínimo si está MUY atascado
        if self.intentos_fallidos > 8:
            # Buscar posición libre cercana
            for radio in [0.5, 1.0, 1.5]:
                for angulo in range(0, 360, 45):
                    rad = np.radians(angulo)
                    test_x = self.x + radio * np.cos(rad)
                    test_y = self.y + radio * np.sin(rad)
                    
                    if not maze.check_collision(test_x, test_y, self.radius):
                        print(f"⚠️  Agente desatascado: {self.intentos_fallidos} intentos")
                        self.x = test_x
                        self.y = test_y
                        self.history.append((self.x, self.y))
                        self.intentos_fallidos = 0
                        return True
            
            # Resetear contador para evitar bucle infinito
            self.intentos_fallidos = 0

        # Si todo falla, quedarse quieto (pero registrar posición)
        self.history.append((self.x, self.y))
        return False


# ===========================================================
#   PERSEGUIDOR (con salto único por episodio)
# ===========================================================

class Perseguidor(Agent):

    def __init__(self, x, y, radius=0.3, speed=1.0):
        super().__init__(x, y, radius, speed)
        self.salto_disponible = True
        self.saltos_usados = 0

    def reset_salto(self):
        """Resetea el salto disponible."""
        self.salto_disponible = True

    def salto(self, evasor_pos, maze):
        """
        Realiza un salto hacia el evasor.
        🔥 MEJORADO: Salto más corto pero más seguro
        """
        if not self.salto_disponible:
            self.history.append((self.x, self.y))
            return False

        ex, ey = evasor_pos
        dx = ex - self.x
        dy = ey - self.y

        distancia = np.sqrt(dx**2 + dy**2)
        
        if distancia < 0.5:  # Ya está muy cerca
            self.history.append((self.x, self.y))
            return False

        # Normalizar dirección
        dx_norm = dx / distancia
        dy_norm = dy / distancia

        # 🔥 SALTO ADAPTATIVO según distancia
        if distancia < 3.0:
            salto_dist = 1.5  # Salto corto
        elif distancia < 6.0:
            salto_dist = 2.5  # Salto medio
        else:
            salto_dist = 3.5  # Salto largo

        new_x = self.x + salto_dist * dx_norm
        new_y = self.y + salto_dist * dy_norm

        # Verificar que el salto no choque
        if maze.check_collision(new_x, new_y, self.radius):
            # 🔥 NUEVO: Intentar salto más corto
            for factor in [0.7, 0.5, 0.3]:
                test_x = self.x + salto_dist * factor * dx_norm
                test_y = self.y + salto_dist * factor * dy_norm
                
                if not maze.check_collision(test_x, test_y, self.radius):
                    self.x = test_x
                    self.y = test_y
                    self.history.append((self.x, self.y))
                    self.salto_disponible = False
                    self.saltos_usados += 1
                    return True
            
            # Salto completamente bloqueado
            self.history.append((self.x, self.y))
            return False

        # Salto exitoso
        self.x = new_x
        self.y = new_y
        self.history.append((self.x, self.y))
        self.salto_disponible = False
        self.saltos_usados += 1
        return True


# ===========================================================
#   EVASOR (coloca trampa temporal)
# ===========================================================

class Evasor(Agent):

    def __init__(self, x, y, radius=0.3, speed=1.0):
        super().__init__(x, y, radius, speed)
        self.trampa_activa = None
        self.trampas_puestas = 0
        self.turnos_trampa = 0

    def poner_trampa(self):
        """Crea una trampa en la posición actual del evasor."""
        self.trampa_activa = (self.x, self.y)
        self.trampas_puestas += 1
        self.turnos_trampa = 0
        return self.trampa_activa

    def actualizar_trampa(self):
        """La trampa desaparece después de 10 turnos."""
        if self.trampa_activa is not None:
            self.turnos_trampa += 1
            if self.turnos_trampa > 10:
                self.trampa_activa = None
                self.turnos_trampa = 0
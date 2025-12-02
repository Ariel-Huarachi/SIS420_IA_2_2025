import numpy as np

class Cazador:
    """Agente cazador que persigue al siguiente en ciclo."""

    def __init__(self, x, y, id=0, radius=0.3, speed=1.0):
        self.x = float(x)
        self.y = float(y)
        self.id = id  # 0, 1, o 2
        self.radius = radius
        self.speed = speed
        self.base_speed = speed
        self.history = [(self.x, self.y)]
        
        # Detección de atasco
        self.intentos_fallidos = 0
        self.ultima_direccion = None
        
        # Trampa
        self.trampa_activa = None
        self.trampas_puestas = 0
        self.turnos_trampa = 0

    def get_position(self):
        return (self.x, self.y)

    def reset_history(self):
        self.history = [(self.x, self.y)]
        self.intentos_fallidos = 0

    def restore_speed(self):
        """Restaurar velocidad base."""
        self.speed = self.base_speed

    def poner_trampa(self):
        """Coloca una trampa en la posición actual."""
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

    # ===========================================================
    #   SISTEMA ANTI-ATASCOS MEJORADO
    # ===========================================================
    def move_angle(self, angle_deg, maze):
        """
        Movimiento continuo con anti-atascos agresivo
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
            self.intentos_fallidos = 0
            self.ultima_direccion = angle_deg
            return True

        # --- INTENTOS ALTERNATIVOS ---
        
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

        # INTENTO 5: Ángulos cercanos (±45°)
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

        # INTENTO 6: Sistema de empuje agresivo
        self.intentos_fallidos += 1
        
        if self.intentos_fallidos > 3:
            empuje = 0.2
            
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
        
        # INTENTO 7: Teletransporte mínimo si está muy atascado
        if self.intentos_fallidos > 8:
            for radio in [0.5, 1.0, 1.5]:
                for angulo in range(0, 360, 45):
                    rad = np.radians(angulo)
                    test_x = self.x + radio * np.cos(rad)
                    test_y = self.y + radio * np.sin(rad)
                    
                    if not maze.check_collision(test_x, test_y, self.radius):
                        self.x = test_x
                        self.y = test_y
                        self.history.append((self.x, self.y))
                        self.intentos_fallidos = 0
                        return True
            
            self.intentos_fallidos = 0

        self.history.append((self.x, self.y))
        return False
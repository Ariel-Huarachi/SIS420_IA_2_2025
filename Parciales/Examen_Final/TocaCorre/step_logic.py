import numpy as np

# ==========================================================
#   MOVER AGENTE SEGÚN ACCIÓN
# ==========================================================

def mover_agente(agent, accion, maze):
    """
    Mueve un agente según acción discreta (0-7 = direcciones)
    """
    if accion < 0 or accion > 7:
        return False
    
    pos_antes = agent.get_position()
    
    angle_deg = accion * 45
    exito = agent.move_angle(angle_deg, maze)
    
    pos_despues = agent.get_position()
    
    distancia_movida = np.linalg.norm([pos_despues[0] - pos_antes[0],
                                       pos_despues[1] - pos_antes[1]])
    
    if distancia_movida < 0.05 and exito:
        direccion_alternativa = (accion + 2) % 8
        angle_alt = direccion_alternativa * 45
        agent.move_angle(angle_alt, maze)
    
    return exito


# ==========================================================
#   APLICAR ZONAS ESPECIALES
# ==========================================================

def aplicar_zonas(agent, env):
    """
    Modifica la velocidad del agente en zonas especiales
    """
    x, y = agent.get_position()
    en_zona_especial = False

    # Zonas rápidas
    for zx, zy, boost in env.zonas_acelerar:
        if np.linalg.norm([x - zx, y - zy]) < 1.0:
            agent.speed = agent.base_speed * boost
            en_zona_especial = True
            return

    # Zonas lentas
    for zx, zy, slow in env.zonas_lentas:
        if np.linalg.norm([x - zx, y - zy]) < 1.0:
            agent.speed = agent.base_speed * slow
            en_zona_especial = True
            return

    if not en_zona_especial:
        agent.restore_speed()


# ==========================================================
#   DETECTAR TOCA - 3 AGENTES
# ==========================================================

def detectar_toca_3agentes(agentes):
    """
    Detecta si cada cazador atrapó a su presa.
    Retorna [toco_0, toco_1, toco_2]
    """
    toques = [False, False, False]
    
    for i in range(3):
        cazador = agentes[i]
        presa = agentes[(i + 1) % 3]
        
        dist = np.linalg.norm([
            cazador.x - presa.x,
            cazador.y - presa.y
        ])
        
        distancia_contacto = cazador.radius + presa.radius + 0.2
        
        if dist < distancia_contacto:
            toques[i] = True
    
    return toques


# ==========================================================
#   APLICAR TRAMPA - 3 AGENTES
# ==========================================================

def aplicar_trampa_3agentes(env):
    """
    Ralentiza al perseguidor si pisa la trampa del cazador.
    Retorna lista de booleanos [en_trampa_0, en_trampa_1, en_trampa_2]
    """
    en_trampa = [False, False, False]
    
    for i in range(3):
        cazador = env.agentes[i]
        perseguidor = env.agentes[(i - 1) % 3]
        
        # Actualizar trampa del cazador
        cazador.actualizar_trampa()
        
        # Si hay trampa activa y el perseguidor la pisa
        if cazador.trampa_activa is not None:
            tx, ty = cazador.trampa_activa
            px, py = perseguidor.get_position()
            
            dist = np.linalg.norm([px - tx, py - ty])
            
            if dist < 0.7:
                perseguidor.speed = perseguidor.base_speed * 0.3
                en_trampa[(i - 1) % 3] = True
            else:
                perseguidor.restore_speed()
    
    return en_trampa


# ==========================================================
#   DETECCIÓN DE ATASCO GLOBAL
# ==========================================================

def detectar_atasco_global(agentes, ventana=20):
    """
    Detecta si algún agente está atascado
    """
    for agente in agentes:
        if len(agente.history) < ventana:
            continue
        
        pos = agente.history[-ventana:]
        varianza_x = np.var([p[0] for p in pos])
        varianza_y = np.var([p[1] for p in pos])
        
        umbral = 0.01
        
        if varianza_x < umbral and varianza_y < umbral:
            return True
    
    return False


def desatascar_agentes(agentes):
    """
    Mueve ligeramente a los agentes si están atascados
    """
    for agente in agentes:
        if agente.intentos_fallidos > 5:
            agente.x += np.random.uniform(-0.5, 0.5)
            agente.y += np.random.uniform(-0.5, 0.5)
            agente.intentos_fallidos = 0
            print(f"⚠️  Agente {agente.id} desatascado")
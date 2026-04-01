import os
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo

def solve_kepler(M_rad, e, tol=1e-6):
    # Iteracyjne rozwiązanie równania Keplera. 
    # Niezbędne do zamiany czasu (Anomalii Średniej) na pozycję (Anomalię Ekscentryczną).
    E = M_rad
    for _ in range(100):
        diff = E - e * np.sin(E) - M_rad
        if abs(diff) < tol:
            break
        E = E - diff / (1.0 - e * np.cos(E) + 1e-8) 
    return E

def get_position_at_time(a, e, i_deg, om_deg, w_deg, ma_deg, n_deg, epoch, t):
    # Wyznaczanie pozycji 3D w układzie heliocentrycznym ekliptycznym.
    # t - epoch: czas jaki upłynął od momentu pomiaru parametrów orbity.
    if e >= 1.0 or n_deg == 0:
        return np.array([np.nan, np.nan, np.nan])
        
    M_deg = (ma_deg + n_deg * (t - epoch)) % 360
    M_rad = np.radians(M_deg)
    
    E = solve_kepler(M_rad, e)
    
    x_orb = a * (np.cos(E) - e)
    y_orb = a * np.sqrt(1 - e**2) * np.sin(E)
    z_orb = 0.0
    
    i, om, w = np.radians(i_deg), np.radians(om_deg), np.radians(w_deg)
    
    R_w = np.array([[np.cos(w), -np.sin(w), 0], [np.sin(w), np.cos(w), 0], [0, 0, 1]])
    R_i = np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]])
    R_om = np.array([[np.cos(om), -np.sin(om), 0], [np.sin(om), np.cos(om), 0], [0, 0, 1]])
    
    R = R_om @ R_i @ R_w
    return R @ np.array([x_orb, y_orb, z_orb])

def generate_orbit_points(a, e, i_deg, om_deg, w_deg, num_points=200):
    i, om, w = np.radians(i_deg), np.radians(om_deg), np.radians(w_deg)
    
    if e >= 1.0:
        nu_limit = np.arccos(-1.0 / e) if e > 1.0 else np.pi
        nu = np.linspace(-nu_limit + 0.1, nu_limit - 0.1, num_points)
        p = np.abs(a) * (e**2 - 1.0)
    else:
        nu = np.linspace(0, 2 * np.pi, num_points)
        p = a * (1.0 - e**2)
        
    r = p / (1.0 + e * np.cos(nu))
    coords = np.vstack((r * np.cos(nu), r * np.sin(nu), np.zeros_like(nu)))
    
    R_w = np.array([[np.cos(w), -np.sin(w), 0], [np.sin(w), np.cos(w), 0], [0, 0, 1]])
    R_i = np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]])
    R_om = np.array([[np.cos(om), -np.sin(om), 0], [np.sin(om), np.cos(om), 0], [0, 0, 1]])
    
    return (R_om @ R_i @ R_w) @ coords

def visualize_animated_neos(neo_list, epoch_start=2459000.5, days_to_simulate=365, frames_count=180):
    
    output_dir = "animations"
    base_filename = "neo_blackbox_orbits_animated"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder created: {output_dir}")

    counter = 1
    while True:
        filename = f"{base_filename}_{counter}.html"
        full_path = os.path.join(output_dir, filename)
        if not os.path.exists(full_path):
            break
        counter += 1
    
    print(f"Visualization generation (frames: {frames_count}) for {len(neo_list)} NEOs in File: {full_path}")
    fig = go.Figure()
    
    times = np.linspace(epoch_start, epoch_start + days_to_simulate, frames_count)
    colors_cycle = ['#FF8C00', '#9370DB', '#32CD32', '#FF1493', '#4169E1', '#8A2BE2', '#FF69B4', '#008080', '#20B2AA', '#4B0082']
    
    # warstwy statyczne
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', 
                               marker=dict(size=14, color='#FFD700', symbol='diamond', line=dict(color='orange', width=2)), 
                               name='Sun', showlegend=True))

    xe, ye, ze = generate_orbit_points(1.0, 0.0167, 0.0, 0.0, 288.1)
    fig.add_trace(go.Scatter3d(x=xe, y=ye, z=ze, mode='lines', 
                               line=dict(color='rgba(0, 191, 255, 0.6)', width=3), 
                               name='Earth trajectory', hoverinfo='skip', showlegend=True))
    
    for i, neo in enumerate(neo_list):
        a, e, inc, om, w = neo['params']
        name = neo['name']
        color = colors_cycle[i % len(colors_cycle)]
        
        xn, yn, zn = generate_orbit_points(a, e, inc, om, w)
        valid = ~np.isnan(xn) & ~np.isinf(xn)
        
        fig.add_trace(go.Scatter3d(x=xn[valid], y=yn[valid], z=zn[valid], mode='lines', 
                                   line=dict(width=2, color=color, dash='dash'), 
                                   name=f"{name}", legendgroup=name, hoverinfo='skip', showlegend=True))

    num_static_traces = 2 + len(neo_list)

    # --- ETAP 2: WARSTWY DYNAMICZNE ---
    dynamic_traces_indices = []
    current_idx = num_static_traces

    # Ziemia (Punkt startowy)
    ex0, ey0, ez0 = get_position_at_time(1.0, 0.0167, 0.0, 0.0, 288.1, 0.0, 0.9856, epoch_start, times[0])
    fig.add_trace(go.Scatter3d(x=[ex0], y=[ey0], z=[ez0], mode='markers', 
                               marker=dict(size=8, color='blue', line=dict(color='white', width=2)), 
                               name='Earth', showlegend=True))
    dynamic_traces_indices.append(current_idx)
    current_idx += 1

    # Wskaźnik Zbliżenia (Pusty na start)
    fig.add_trace(go.Scatter3d(x=[ex0, ex0], y=[ey0, ey0], z=[ez0, ez0], mode='lines', 
                               line=dict(color='red', width=3, dash='dot'), 
                               name='Line of Approach', showlegend=False))
    dynamic_traces_indices.append(current_idx)
    current_idx += 1

    # Punkty NEO (Startowe z obsługą tekstu)
    for i, neo in enumerate(neo_list):
        a, e, inc, om, w = neo['params']
        ma, n = neo['time_params']['ma'], neo['time_params']['n']
        nx0, ny0, nz0 = get_position_at_time(a, e, inc, om, w, ma, n, epoch_start, times[0])
        name = neo['name']
        
        # Ustawiamy mode na 'markers+text', żeby w klatkach animacji móc dynamicznie dopisywać ostrzeżenia
        fig.add_trace(go.Scatter3d(x=[nx0], y=[ny0], z=[nz0], mode='markers+text', 
                                   marker=dict(size=6, symbol='circle', color='gray'), 
                                   text=[""], textposition="top right",
                                   name=f"Position: {name}", legendgroup=name, showlegend=False))
        dynamic_traces_indices.append(current_idx)
        current_idx += 1


    # --- ETAP 3: GENEROWANIE KLATEK ---
    frames = []
    AU_TO_LD = 389.17

    for step, t in enumerate(times):
        frame_data = [] 
        
        # 1. Pozycja Ziemi
        ex, ey, ez = get_position_at_time(1.0, 0.0167, 0.0, 0.0, 288.1, 0.0, 0.9856, epoch_start, t)
        frame_data.append(go.Scatter3d(x=[ex], y=[ey], z=[ez])) 

        closest_neo_dist = float('inf')
        closest_neo_pos = (ex, ey, ez)
        neo_points_data = [] 
        
        # 2. Pozycje NEO (Z logiką CLOSE APPROACH)
        for neo in neo_list:
            a, e, inc, om, w = neo['params']
            ma, n = neo['time_params']['ma'], neo['time_params']['n']
            nx, ny, nz = get_position_at_time(a, e, inc, om, w, ma, n, epoch_start, t)
            
            dist_au = np.sqrt((nx-ex)**2 + (ny-ey)**2 + (nz-ez)**2)
            dist_ld = dist_au * AU_TO_LD
            
            # --- LOGIKA ZMIANY KOLORÓW I OSTRZEŻEŃ ---
            if dist_ld <= 15.0: # ALARM - Czerwony (< 15 LD)
                marker_color = '#FF0000' # Mocny czerwony
                marker_size = 12
                alert_text = f"Warning {neo['name']} ({dist_ld:.1f} LD)"
                
                if dist_ld < closest_neo_dist:
                    closest_neo_dist = dist_ld
                    closest_neo_pos = (nx, ny, nz)
                    
            elif dist_ld <= 50.0: # OSTRZEŻENIE - Pomarańczowy (< 50 LD)
                marker_color = '#FF8C00' # Ciemny pomarańcz
                marker_size = 9
                alert_text = "" # Brak tekstu, tylko większa kropka
                
                if dist_ld < closest_neo_dist:
                    closest_neo_dist = dist_ld
                    closest_neo_pos = (nx, ny, nz)
                    
            else: # BEZPIECZNIE - Zielony
                marker_color = 'rgba(0, 150, 0, 0.7)' # Przytłumiony zielony
                marker_size = 5
                alert_text = ""

            neo_points_data.append(go.Scatter3d(
                x=[nx], y=[ny], z=[nz], 
                mode='markers+text', # Wymagane do wyświetlania alertów
                text=[alert_text],
                textfont=dict(color='red', size=14, family="Arial Black"),
                textposition="middle right",
                marker=dict(size=marker_size, color=marker_color, line=dict(width=1, color='black') if dist_ld<=15 else None),
                hovertemplate=f"<b>{neo['name']}</b><br>Distance: {dist_ld:.1f} LD<br>Data: Day {int(t-epoch_start)}<extra></extra>"
            ))

        # 3. Wskaźnik zbliżenia (czerwona linia) - pojawia się, gdy jakikolwiek obiekt wejdzie w strefę < 50 LD
        if closest_neo_dist <= 50.0:
            frame_data.append(go.Scatter3d(x=[ex, closest_neo_pos[0]], y=[ey, closest_neo_pos[1]], z=[ez, closest_neo_pos[2]]))
        else:
            frame_data.append(go.Scatter3d(x=[ex, ex], y=[ey, ey], z=[ez, ez])) 
            
        frame_data.extend(neo_points_data)
        frames.append(go.Frame(data=frame_data, name=str(step), traces=dynamic_traces_indices))

    fig.frames = frames

    # --- ETAP 4: UKŁAD ---
    fig.update_layout(
        title='Black Box NEO Orbit Analysis: Close approach warning',
        paper_bgcolor='white', 
        scene=dict(
            xaxis=dict(title='X (AU)', range=[-3, 3], backgroundcolor="white", gridcolor="rgb(200, 200, 200)", showbackground=True),
            yaxis=dict(title='Y (AU)', range=[-3, 3], backgroundcolor="white", gridcolor="rgb(200, 200, 200)", showbackground=True),
            zaxis=dict(title='Z (AU)', range=[-1, 1], backgroundcolor="white", gridcolor="rgb(200, 200, 200)", showbackground=True),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
            bgcolor='white' 
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            font=dict(color="black"), 
            bgcolor="rgba(255,255,255,0.8)",
            itemsizing='constant'
        ),
        updatemenus=[dict(type="buttons",
                          font=dict(color="black"),
                          buttons=[dict(label="▶ Start", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                                   dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(
            font=dict(color="black"),
            steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=50, redraw=True))], label=f"Day {int(times[k]-epoch_start)}") for k in range(frames_count)])]
    )
    
    pyo.plot(fig, filename=full_path, auto_open=True)
    print(f"Saved animation in: {full_path}")
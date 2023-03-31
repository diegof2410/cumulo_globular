function [R,p,Estrellas] = contador_cumulo_2(img,n_a,r,u_b,u_pb,X,O)
%La función contador_cumulo es una función que analiza imagenes de cumulos 
%globulares. Toma 6 argumentos de
%entrada.
%
%img: Imagen a analizar, debe escribirse como 'nombreimagen.extension'.
%n_a: Número de anillos deseados para reliazar el estudio.
%r: Radio de discriminación para las estrellas grandes.
%u_b: Umbral para encontrar estrellas brillantes.
%u_pc: Umbral para encontrar estrellas poco brillantes, u_b y u_pb actúan 
% como cotas para ese caso.
%X: Disancia de la tierra al cúmulo (PC)
%O: Ángulo con el que se obtuvo la imagen (radianes).
%
%Entregando como resultado
%
%R: Un vector de posiciones que mide desde el
%centro del cumulo hasta la posición media del anillo en el que se mide la
%densidad (pc). 
%p: Vector que contiene la densidad superficial de estrellas en cada anillo 
% (pc^2).
%Estrellas: el número total de estrellas detectadas en la imagen. 
%
%También muestra las gráficas de la densidad superficial y el número de
%estrellas en función de la distancia al centro del cúmulo, la imagen original a 
% escala de grises, 
%la imagen construida por el algoritmo empleado, la imagen construida con los 
%anillos deseados superpuestos y la gráfica del mejor ajuste usando la 
% ecuación de King. 

%Hacemos robusta la función

%Verificamos que haya suficientes argumentos de entrada

if nargin < 7
    error(['La función requiere exactamente 7 argumentos de entrada.' ...
        ' Para conocer más sobre ellos utilice el comando "help contador_cumulo".']);
end

%Verificamos que existan dichos argumentos
if ~exist('img','var')
    error('El argumento (1) de imagen no está definido.')
end
if ~exist('n_a','var')
    error('El argumento (2) de numero de anillos no está definido.')
end
if ~exist('r','var')
    error('El argumento (3) de radio de discriminación no está definido.')
end
if ~exist('u_b','var')
    error('El argumento (4) de umbral superior no está definido.')
end
if ~exist('u_pb','var')
    error('El argumento (5) de umbral inferior no está definido.')
end
if ~exist('X','var')
    error('El argumento (6) de distancia no está definido.')
end
if ~exist('O','var')
    error('El argumento (7) de ángulo no está definido.')
end

%Comprobamos que sean del tipo de datos deseados
ar_num = {n_a,r,u_b,u_pb,X,O};

condicion = false(1,length(ar_num));
if ischar(img) || length(size(img)) >= 2
    for ii = 1:length(ar_num)
        if isnumeric(ar_num{ii})
            if ii == 1
                if all(round(ar_num{ii}) == ar_num{ii}) && all(ar_num{ii} > 0)
                    if any(ar_num{ii} < 5)
                        error('El número mínimo de anillos es 5.')
                    else
                        condicion(ii) = true;
                    end
                else
                    error(['El número de anillos y radio de discriminación ' ...
                        'deben ser enteros mayores que 0.']);
                end
            elseif ii == 2
                if round(ar_num{ii}) == ar_num{ii} && ar_num{ii} > 0
                    condicion(ii) = true;
                else
                    error(['El número de anillos y radio de discriminación ' ...
                        'deben ser enteros mayores que 0.']);
                end                
            elseif ii == 3 || ii == 4
                if round(ar_num{ii}) == ar_num{ii} && ar_num{ii} >= 0 && ar_num{ii} <= 255
                    condicion(ii) = true;
                else
                    error('Los umbrales deben tener valores enteros entre 0 y 255.')
                end
            else
                condicion(ii) = true;
            end
        end
    end

    if condicion
        %Se ejecuta el código si se cumplen las condiciones necesarias.

        %Cargamos la imagen.
        if ischar(img)
            cumulo = imread(img);
        else
            cumulo = img;
        end
        dimensiones = size(cumulo);

        %Comprobamos si es una imagen a escala de grises o en rgb.
        if length(dimensiones) == 2
            for ii = 1:dimensiones(1)
                for jj = 1:dimensiones(2)
                    if cumulo(ii,jj)>= 0 && cumulo(ii,jj)<=255
                        [row,col] = size(cumulo);
                    end
                end
            end
        else
            cumulo = rgb2gray(cumulo);
            [row,col] = size(cumulo);
        end

        %Establecemos el origen de los anillos concentricos.
        col_0 = round(col / 2);
        row_0 = round(row / 2);

        %Generamos matrices de las capas de imagen.
        cumulo_EPB = false(row,col); %estrellas poco brillantes
        cumulo_construida = false(row,col); %Imagen construida

        %Encontramos las estrelllas

        %Estrellas grandes/brillantes
        for ii = 1:row
            for jj = 1:col
                if all([ii, jj] > r) && ii < row-r && jj < col-r
                    %Realiza el proceso en el interior.
                    if cumulo(ii-r:ii+r, jj-r:jj+r) >= u_b
                        %Copia estrellas grandes.
                        cumulo_construida(ii-r:ii+r, jj-r:jj+r) = true;
                    elseif cumulo(ii,jj) >= u_b
                        %Copia estrellas pequenas pero brillantes.
                        cumulo_construida(ii,jj) = true;
                    end
                elseif cumulo(ii,jj) >= u_b 
                    %Realiza el proceso en las orillas.
                    cumulo_construida(ii,jj) = true;
                end
            end
        end

        %Estrellas con pixeles entre ambos umbrales.
        for ii = 1:row
            for jj = 1:col
                if cumulo(ii,jj) > u_pb && cumulo(ii,jj) < u_b
                    cumulo_EPB(ii,jj) = true;
                end
            end
        end
        %algunos de los pixeles son estrellas, otros son resplandor de las mas
        %brillantes y tendrán un hueco en medio.

        %Número de pixeles mínimo para que no haya agujero.
        n = (2*r+1)^2-1;

        %Eliminamos el ruido
        ruido_discriminado = imfill(cumulo_EPB,'holes');
        ruido_discriminado = bwareafilt(ruido_discriminado,[0 n]);

        %Anadimos estrellas pequeñas a la imagen construida.
        for ii = 1:row
            for jj= 1:col
                if ruido_discriminado(ii,jj)
                    cumulo_construida(ii,jj) = true;
                end
            end
        end

        %Etiquetamos los objetos en la imagen.
        [L, Estrellas] = bwlabel(cumulo_construida);
        %L es la matriz locations, indica las coordenadas        

        %Asignamos un espacio para los cálculos en cada anillo
        anillosT = length(n_a);
        anillos = n_a;

        RSQ = zeros(1,anillosT);
        RR = cell(1:anillosT);
        pp = cell(1:anillosT);
        r_C = zeros(1,anillosT);

        %Calculamos en cada anillo
        for zz = 1:anillosT
            n_a = anillos(zz);
            %Definimos la distancia entre radios de anillos sucesivos.
            if col <= row
                n_amax = length(col_0:col)/2 - 1;
                %Comprobamos se pueda establecer ese número de anillos.
                if n_a > n_amax
                    error(['El número de anillos deseados sobrepasa al número de anillos' ...
                        ' posibles %d.'], n_amax);
                end
                multiplicador = floor(length(col_0:col) / (n_a));
            else
                n_amax = length(row_0:row) - 1;
                %Comprobamos se pueda establecer ese número de anillos.
                if n_a > n_amax
                    error(['El número de anillos deseados sobrepasa al número de anillos' ...
                        ' posibles %d.'], n_amax);
                end
                multiplicador = floor(length(row_0:row) / (n_a));
            end
    
            %Definimos los diferentes radios.
            r_a = zeros(1, n_a);
            for ii = 1:n_a
                r_a(ii) = multiplicador * ii;
            end
    
            %Preasignamos vector de estrellas por anillo
            estrellas_por_anillo = zeros(1,n_a);
    
            % Encontrar el centro de cada objeto
            for k = 1:Estrellas
                % Encontrar los índices de los píxeles etiquetados con el valor k
                [r, c] = find(L == k);
        
                % Encontrar el píxel central
                r_centro = round(mean(r));
                c_centro = round(mean(c));
        
                %Determinar la distancia del centro a cada anillo
                d = sqrt((r_centro - row_0)^2 + (c_centro - col_0)^2);
    
                %Buscamos en que anillo esta el objeto
                anillo = 1; %anillo inicial
                while d > r_a(anillo)
                    anillo = anillo + 1;
                    %contamos hasta el ultimo anillo
                    if anillo > n_a
                        break;
                    end
                end
        
                %contamos las estrellas dentro del anillo
                if anillo <= n_a
                    estrellas_por_anillo(anillo) = estrellas_por_anillo(anillo) + 1;
                end
            end
    
            %Distancia promedio
            R = zeros(1,n_a);
            for ii = 1:n_a
                if ii == 1
                    R(ii) = r_a(ii)/2;
                else
                    R(ii) = (r_a(ii) + r_a(ii-1))/2;
                end
            end
    
            %Establecemos las areas de los anillos y convertimos px-pc
    
            %Altura de la imagen en la realidad (Aproximacion tanp=sinp).
            Y = X*sin(O);
            %Equivalencia de la longitud de un pixel.
            l_px = Y/row;
            % Area de un pixel.
            A_px = l_px^2;
    
            % calculamos area de los anillos.
            A = A_px * pi * r_a.^2;
            A_a = zeros(1,length(A));
    
            for ii = 1:length(A)
                if ii == 1
                    A_a(ii) = A(ii);
                else
                    A_a(ii) = A(ii) - A(ii-1);
                end
            end
    
            %Calculamos la densidad superficial de estrellas. 
            p = estrellas_por_anillo./A_a;
    
            %Ajustamos los datos obtenidos
            p0 = p(1);
    
            %Ecuación de ajuste
            densidad_king = @(rc,r) p0*(1+(r./rc).^2).^(-3/2);
    
            rc_optimo = lsqcurvefit(densidad_king, max(R), R, p);
    
            % Cálculo de coeficiente de determinación (R^2)
            p_king = densidad_king(rc_optimo, R);
            rsq = 1 - sum((p - p_king).^2)/sum((p - mean(p)).^2);

            %Asignamos los valores de cada solución

            RSQ(zz) = rsq;
            RR{zz} = R;
            pp{zz} = p;
            r_C(zz) = rc_optimo;
        end

        %Elegimos el mejor ajuste de los realizados.

        for ii = 1:length(RSQ)
            if RSQ(ii) == max(RSQ)
                index = ii;
            end
        end

        R = RR{index};
        p = pp{index};
        p0 = p(1);
        rsq = RSQ(index);
        rking = r_C(index);
        anillo = anillos(index);

        %Ecuación de King
        densidad_king = @(rc,r) p0*(1+(r./rc).^2).^(-3/2);
        %p_king = densidad_king(rking, R);

        %Hacemos la imagen de anillos.
        anillos_img = zeros(row,col);

        rc = zeros(1,anillo);
        rc_counter = 0;
        for ii = 1:anillo
            rc_counter = rc_counter + 1;
            rc(ii) = floor(col/(2*anillo)) * rc_counter;
        end

        col_0 = round(col/2);
        row_0 = round(row/2);

        circunferencias = zeros(row, col, anillo);
        for kk = 1:anillo
            for ii = 1:row
                for jj = 1:col
                    if (ii-row_0)^2 + (jj-col_0)^2 <= rc(kk)^2
                        anillos_img(ii,jj) = true;
                    end
                end
            end
            circunferencias(:,:,kk) = bwperim(anillos_img);
        end

        circunferencia = sum(circunferencias, 3) > 0;
        cumulo_construida_anillos = cumulo_construida;

        for ii = 1:row
            for jj = 1:col
                if circunferencia(ii,jj)
                    cumulo_construida_anillos(ii,jj) = true;
                end
            end
        end       

        figure(1); %Datos experimentales y ajuste
        scatter(R,p,20,'filled','MarkerFaceColor','b');
        hold on;
        r = linspace(min(R),max(R),150);
        plot(r,densidad_king(rking,r),'r-','LineWidth',2);
        grid on
        xlabel('Distancia desde el centro del cúmulo (PC)', 'FontSize', 14, ...
            'FontName', 'Times')
        ylabel('Densidad estelar $\frac{estrellas}{pc^2}$', 'FontSize', 14, ...
            'FontName', 'Times', 'Interpreter', 'latex')
        legend({'Datos experimentales', 'Mejor ajuste teórico'}, 'FontSize', ...
            14, 'FontName', 'Times')
        titleStr = sprintf('Ajuste: ecuación de King para %d anillos. R^2 = %s', anillo, num2str(rsq, '%.3f'));
        title(titleStr, 'FontSize', 16, 'FontName', 'Times');        
        hold off

        figure(2) %Imagen original
        imshow(cumulo)

        figure(3) %Imagen construida de ruido reducido
        imshow(cumulo_construida)

        figure(4) %Imagen construida y anillos 
        imshow(cumulo_construida_anillos)

    else
        error('Los argumentos 2-7 deben ser de tipo numérico')
    end
else
    error(['El nombre de la imagen debe estar escrito entre comillas, o' ...
        ' asignado a una variable']);
end
end